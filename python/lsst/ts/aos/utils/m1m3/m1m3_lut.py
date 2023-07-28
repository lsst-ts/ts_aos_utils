# This file is part of ts_aos_utils.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import asyncio
import enum
import os

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time
from lsst.ts.criopy import M1M3FATable
from lsst_efd_client import EfdClient
from numpy.polynomial import Polynomial
from tqdm import tqdm


class ForceType(enum.IntEnum):
    BALANCE = 1
    APPLIED = 2


async def retrieve_elevations(
    start_time: Time, end_time: Time, resample_rate: float, client: EfdClient
) -> pd.DataFrame:
    """Retrieve elevations data.

    Parameters
    ----------
    start_time: astropy.time.Time
        Start time of the data to be queried from the EFD
    end_time: astropy.time.Time
        End time of the data to be queried from the EFD
    resample_rate: str
        Rate at which the data is resampled. Default is '1T' (1 minute)
    client: EfdClient
        Client for EFD database

    Returns
    -------
    data: pd.DataFrame
        Resampled elevation data.
    """

    interval_ms = (end_time - start_time).to(u.ms).value

    # Query elevation from EFD from start_time to end_time
    print("Retrieving elevation data from EFD...")
    elevations = await client.select_time_series(
        "lsst.sal.MTMount.elevation",
        ["actualPosition", "timestamp"],
        start_time,
        end_time,
    )
    print("Elevations\n", elevations["actualPosition"].describe())
    expected_count = interval_ms / 70.0
    if len(elevations.index) < expected_count:
        print(
            f"WARNING: There is significant difference between expected {expected_count:.0f} "
            f"and retrived {len(elevations.index):d}  - PROBABLY ELEVATION DATA ARE MISSING?"
        )

    # Resample the data
    elevations = elevations["actualPosition"].resample(resample_rate).mean()

    if min(elevations) > 20 or max(elevations) < 70:
        # Add warning
        print(
            "WARNING: Elevation range is not enough for coefficient fitting - shall be in 20-70 degrees range"
        )
    return elevations


async def update_lut_force_balance(
    force_type: ForceType,
    start_time: Time,
    end_time: Time,
    axis: str,
    elevations: pd.DataFrame,
    lut_path: None | str,
    polynomial_degree: int,
    resample_rate: float,
    client: EfdClient,
) -> None:
    """Update the LUT file with the force balance data from the EFD.
    Saves a new .csv file with LUT updated values

    Parameters
    ----------
    force_type: ForceType
        Type of force to be updated. ForceType.BALANCE or ForceType.APPLIED
    start_time: astropy.time.Time
        Start time of the data to be queried from the EFD
    end_time: astropy.time.Time
        End time of the data to be queried from the EFD
    axis: str
        Axis of the force balance to be updated. Can be 'X', 'Y', or 'Z'
    elevations: pd.DataFrame
        Elevation data - see retrieve_elevations how toget those.
    lut_path: str
        Path to the LUT file to be updated. Can be None for Applied forces, as
        it isn't used.
    polynomial_degree: int
        Degree of the polynomial to be fitted to the data
    resample_rate: str
        Rate at which the data is resampled. Default is '1T' (1 minute)
    client: EfdClient
        Client for EFD database

    Returns
    -------
    None
    """
    interval_ms = (end_time - start_time).to(u.ms).value
    index_type = getattr(M1M3FATable.FAIndex, axis)
    num_actuators = getattr(M1M3FATable, f"FATABLE_{axis}FA")

    # Get path of the LUT file
    out_file = f"Elevation{axis}Table.csv"

    # Get names of actuator forces
    query_forces = [f"{axis.lower()}Forces{i}" for i in range(num_actuators)]

    # Query appliedBalanceForces from EFD from start_time to end_time
    if force_type == ForceType.BALANCE:
        assert lut_path is not None

        lut_file = os.path.join(lut_path, out_file)
        print("Reading elevation coefficients from file", lut_file)
        table_file = pd.read_csv(lut_file)

        print("Retrieving applied balance forces data from EFD...")
        forces = await client.select_time_series(
            "lsst.sal.MTM1M3.appliedBalanceForces",
            query_forces,
            start_time,
            end_time,
        )
    elif force_type == ForceType.APPLIED:
        table_file = pd.DataFrame(
            {
                "ID": [row.actuator_id for row in M1M3FATable.FATABLE],
                "Coefficient 5": [0] * M1M3FATable.FATABLE_ZFA,
                "Coefficient 4": [0] * M1M3FATable.FATABLE_ZFA,
                "Coefficient 3": [0] * M1M3FATable.FATABLE_ZFA,
                "Coefficient 2": [0] * M1M3FATable.FATABLE_ZFA,
                "Coefficient 1": [0] * M1M3FATable.FATABLE_ZFA,
                "Coefficient 0": [0] * M1M3FATable.FATABLE_ZFA,
            }
        )

        print("Retrieving applied forces data from EFD...")
        forces = await client.select_time_series(
            "lsst.sal.MTM1M3.appliedForces",
            query_forces,
            start_time,
            end_time,
        )

        # Query static forces from efd. Get the last sample that was
        # published before start time.
        query = f"""
            SELECT "{query_forces}"
            FROM "efd"."autogen"."lsst.sal.MTM1M3.logevent_appliedStaticForces"
            WHERE time < '{start_time.isot}+00:00' ORDER BY DESC LIMIT 1
        """
        static_forces = await client.influx_client.query(query)

        for idx in tqdm(range(num_actuators)):
            forces[query_forces[idx]] = forces[query_forces[idx]].subtract(
                static_forces[query_forces[idx]]
            )

    expected_count = interval_ms / 20.0
    if abs(expected_count - len(forces.index)) > 10:
        print(
            "WARNING: Most likely too few hardpoints samples - expected "
            f"{expected_count:.0f}, found {len(forces.index)}"
        )

    # Resample the data to in bins of length resample_bin
    # (this can be in minutes, seconds, etc.)
    forces_resampled = forces.resample(resample_rate).mean()

    print("Fitting forces with polynomial...")

    residuals = []

    # For each actuator update the LUT row
    for row in M1M3FATable.FATABLE:
        idx = row.get_index(index_type)
        if idx is None:
            continue

        new_poly, statistics = Polynomial.fit(
            90 - elevations,
            forces_resampled[query_forces[idx]],
            polynomial_degree,
            full=True,
        )
        res = statistics[0][0]
        print(f"Fit {row.actuator_id} {axis} ({idx:3d}) residuals {res:.5f}")
        residuals.append(res)

        coefs = np.flip(new_poly.convert().coef)

        if force_type == ForceType.BALANCE:
            coefs = np.insert(coefs, 0, 0)
            table_file.loc[table_file["ID"] == row.actuator_id] += coefs
        elif force_type == ForceType.BALANCE:
            coefs = np.insert(coefs, 0, row.actuator_id)
            table_file.loc[table_file["ID"] == row.actuator_id] = coefs

    print(
        f"Residuals {np.min(residuals):.5f} {np.mean(residuals):.5f} {np.max(residuals):.5f}"
    )

    # Save the updated LUT file
    print(f"Saving LUT file {out_file}")
    table_file.to_csv(out_file, index=False)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update M1M3 LUT file with force balance data"
    )
    parser.add_argument(
        "force_type", choices=["Balance", "Applied"], help="Type of forces to be used"
    )
    parser.add_argument(
        "start_time",
        help="Start time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'",
        type=Time,
    )
    parser.add_argument(
        "end_time", help="End time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'", type=Time
    )
    parser.add_argument(
        "--axes",
        choices=["X", "Y", "Z", "XY", "XZ", "YZ", "XZY"],
        default="XYZ",
        help="Axis of the force balance to be updated. Defaults to all axis.",
    )

    parser.add_argument(
        "--lut-path", default=None, help="Path to the LUT file to be updated"
    )

    parser.add_argument(
        "--polynomial-degree",
        default=5,
        type=int,
        help="Degree of the polynomial to be fitted to the data",
    )
    parser.add_argument(
        "--resample-rate",
        default="1T",
        help="Rate at which the data is resampled (default: 1 minute)",
    )
    parser.add_argument(
        "--efd",
        default="usdf_efd",
        help="EFD name. Defaults to usdf_efd",
    )

    return parser.parse_args()


async def fit_tables(args: argparse.Namespace):
    force_type = getattr(ForceType, args.force_type.upper())

    # Initialize EFD client
    print("Conecting to EFD", args.efd)
    client = EfdClient(args.efd)

    elevations = await retrieve_elevations(
        args.start_time, args.end_time, args.resample_rate, client
    )

    if force_type == ForceType.BALANCE:
        if args.lut_path is None:
            print("--lut-path argument, required for Balance forces, is missing")
            return

    for axis in args.axes:
        print(f"Fitting elevation coefficients for {axis} axis")
        # Run the update_lut_force_balance function asynchronously
        await update_lut_force_balance(
            force_type,
            args.start_time,
            args.end_time,
            axis,
            elevations,
            args.lut_path,
            args.polynomial_degree,
            args.resample_rate,
            client,
        )


def run() -> None:
    """Main function to run the update_lut_force_balance function."""
    asyncio.run(fit_tables(parse_arguments()))


if __name__ == "__main__":
    run()
