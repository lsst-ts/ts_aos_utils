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
from astropy.time import Time, TimeDelta
from lsst.ts.criopy import M1M3FATable
from lsst_efd_client import EfdClient
from numpy.polynomial import Polynomial
from tqdm import tqdm


class ForceType(enum.IntEnum):
    BALANCE = 1
    APPLIED = 2


async def update_lut_force_balance(
    force_type: ForceType,
    start_time: Time,
    end_time: Time,
    axis: str,
    lut_path: None | str,
    polynomial_degree: int,
    resample_rate: float,
    static_offset: int = 1,
    efd_name: str = "usdf_efd",
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
    lut_path: str
        Path to the LUT file to be updated
    polynomial_degree: int
        Degree of the polynomial to be fitted to the data
    resample_rate: str
        Rate at which the data is resampled. Default is '1T' (1 minute)
    static_offset: int
        Offset in days to be used when querying the static forces from the EFD.
    efd_name: str
        EFD name.

    Returns
    -------
    None
    """
    interval_ms = (end_time - start_time).to(u.ms).value
    index_type = getattr(M1M3FATable.FAIndex, axis)
    num_actuators = getattr(M1M3FATable, f"FATABLE_{axis}FA")

    # Initialize EFD client
    client = EfdClient("usdf_efd")

    # Get path of the LUT file
    out_file = f"Elevation{axis}Table.csv"
    if lut_path is None:
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
    else:
        lut_file = os.path.join(lut_path, out_file)
        print("Reading", lut_file)
        table_file = pd.read_csv(lut_file)

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
        print("WARNING: Elevation data is not in the range 20-70 degrees")

    # Get names of actuator forces
    query_forces = [f"{axis.lower()}Forces{i}" for i in range(num_actuators)]

    # Query appliedBalanceForces from EFD from start_time to end_time
    if force_type == ForceType.BALANCE:
        print("Retrieving applied balance forces data from EFD...")
        forces = await client.select_time_series(
            "lsst.sal.MTM1M3.appliedBalanceForces",
            query_forces,
            start_time,
            end_time,
        )
    elif force_type == ForceType.APPLIED:
        print("Retrieving applied forces data from EFD...")
        forces = await client.select_time_series(
            "lsst.sal.MTM1M3.appliedForces",
            query_forces,
            start_time,
            end_time,
        )

        # query static forces from efd from start_time - static_offset to
        # start_time
        print("Retrieving static forces data from EFD...")
        static_forces = await client.select_time_series(
            "lsst.sal.MTM1M3.logevent_appliedStaticForces",
            query_forces,
            start_time - TimeDelta(static_offset),
            start_time,
        )
        static_forces = static_forces.iloc[-1]

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
        print(f"Fit {row.actuator_id} {axis} ({idx: 3}) residuals {res:.5f}")
        residuals.append(res)

        coefs = np.flip(new_poly.convert().coef)

        if force_type == ForceType.BALANCE:
            coefs = np.insert(coefs, 0, 0)
            table_file.loc[table_file["ID"] == row.actuator_id] += coefs
        elif force_type == ForceType.APPLIED:
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
        "--lut_path", default=None, help="Path to the LUT file to be updated"
    )

    parser.add_argument(
        "--polynomial_degree",
        default=5,
        type=int,
        help="Degree of the polynomial to be fitted to the data",
    )
    parser.add_argument(
        "--resample_rate",
        default="1T",
        help="Rate at which the data is resampled (default: 1 minute)",
    )
    parser.add_argument(
        "--static_offset",
        default=1 * u.day,
        type=int,
        help="Offset in hours to be used when querying static forces",
    )
    parser.add_argument(
        "--efd",
        default="usdf_efd",
        help="EFD name. Defaults to usdf_efd",
    )

    return parser.parse_args()


def run() -> None:
    """Main function to run the update_lut_force_balance function."""
    args = parse_arguments()

    force_type = getattr(ForceType, args.force_type.upper())
    resample_rate = (
        args.resample_rate
    )  # User-provided resample rate (default: 1 minute)

    for axis in args.axes:
        print("Fitting", axis, "axis")
        # Run the update_lut_force_balance function asynchronously
        asyncio.run(
            update_lut_force_balance(
                force_type,
                args.start_time,
                args.end_time,
                axis,
                args.lut_path,
                args.polynomial_degree,
                resample_rate,
                args.static_offset,
                args.efd,
            )
        )


if __name__ == "__main__":
    run()
