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

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta
from lsst.ts.cRIOpy import M1M3FATable
from lsst_efd_client import EfdClient
from numpy.polynomial import Polynomial
from tqdm import tqdm


class ForceType(enum.IntEnum):
    BALANCE = 1
    APPLIED = 2


async def update_lut_force_balance(
    force_type,
    start_time,
    end_time,
    axis,
    lut_path,
    polynomial_degree,
    resample_rate,
    static_offset=1,
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

    Returns
    -------
    None
    """

    # Initialize EFD client
    client = EfdClient("usdf_efd")

    # Generate array with the ID of each actuator
    fat = np.array(M1M3FATable.FATABLE, dtype=object)
    ids = fat[:, M1M3FATable.FATABLE_ID]

    # Get the number of actuators
    num_actuators = M1M3FATable.FATABLE_ZFA
    axis_indices = fat[:, 11]
    if axis == "X":
        num_actuators = M1M3FATable.FATABLE_XFA
        axis_indices = fat[:, 9]
    elif axis == "Y":
        num_actuators = M1M3FATable.FATABLE_YFA
        axis_indices = fat[:, 10]

    # Get path of the LUT file
    lut_file = os.path.join(lut_path, f"Elevation{axis}Table.csv")
    table_file = pd.read_csv(lut_file)

    # Query elevation from EFD from start_time to end_time
    print("Retrieving elevation data from EFD...")
    elevations = await client.select_time_series(
        "lsst.sal.MTMount.elevation",
        ["actualPosition", "timestamp"],
        start_time,
        end_time,
    )
    # Resample the data
    elevations = elevations["actualPosition"].resample(resample_rate).mean()

    if min(elevations) > 20 and max(elevations) < 70:
        # Add warning
        print("WARNING: Elevation data is not in the range 20-70 degrees")

    print("Retrieving forces data from EFD...")
    # Get names of actuator forces
    query_forces = [f"{axis.lower()}Forces{i}" for i in range(num_actuators)]

    # Query appliedBalanceForces from EFD from start_time to end_time
    if force_type == ForceType.BALANCE:
        forces = await client.select_time_series(
            "lsst.sal.MTM1M3.appliedBalanceForces",
            query_forces,
            start_time,
            end_time,
        )
    elif force_type == ForceType.APPLIED:
        forces = await client.select_time_series(
            "lsst.sal.MTM1M3.appliedForces",
            query_forces,
            start_time,
            end_time,
        )

        # query static forces from efd from start_time - 6h to end_time
        static_forces = await client.select_time_series(
            "lsst.sal.MTM1M3.logevent_appliedStaticForces",
            query_forces,
            start_time - TimeDelta(static_offset),
            end_time,
        )
        static_forces = static_forces.iloc[-1]

        for idx in tqdm(range(num_actuators)):
            forces[query_forces[idx]] = forces[query_forces[idx]].subtract(
                static_forces[query_forces[idx]]
            )

    # Resample the data to in bins of length resample_bin
    # (this can be in minutes, seconds, etc.)
    forces_resampled = forces.resample(resample_rate).mean()

    print("Fitting forces with polynomial...")
    # For each actuator update the LUT row
    for idx in tqdm(range(num_actuators)):
        new_poly = Polynomial.fit(
            90 - elevations, forces_resampled[query_forces[idx]], polynomial_degree
        )
        coefs = np.flip(new_poly.convert().coef)

        actuator_id = ids[np.where(axis_indices == idx)[0][0]]
        if force_type == "Balance":
            coefs = np.insert(coefs, 0, 0)
            table_file.loc[table_file["ID"] == actuator_id] += coefs
        elif force_type == "Applied":
            coefs = np.insert(coefs, 0, actuator_id)
            table_file.loc[table_file["ID"] == actuator_id] = coefs

    # Save the updated LUT file
    print("Saving LUT file...")
    table_file.to_csv(f"Elevation{axis}Table.csv", index=False)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update LUT file with force balance data"
    )
    parser.add_argument(
        "force_type", choices=["Balance", "Applied"], help="Type of forces to be used"
    )
    parser.add_argument(
        "start_time", help="Start time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'"
    )
    parser.add_argument(
        "end_time", help="End time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'"
    )
    parser.add_argument(
        "axis", choices=["X", "Y", "Z"], help="Axis of the force balance to be updated"
    )

    lut_path = f"{os.environ['HOME']}" "/develop/ts_m1m3support/SettingFiles/Tables/"
    parser.add_argument(
        "--lut_path", default=lut_path, help="Path to the LUT file to be updated"
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
        default=1,
        type=int,
        help="Offset in hours to be used when querying static forces",
    )

    return parser.parse_args()


def run() -> None:
    """Main function to run the update_lut_force_balance function."""
    args = parse_arguments()

    # Parse start_time and end_time strings into astropy.time.Time objects
    start_time = Time(args.start_time, format="iso", scale="utc")
    end_time = Time(args.end_time, format="iso", scale="utc")

    force_type = getattr(ForceType, args.force_type.upper())
    axis = args.axis  # User-provided axis (X, Y, or Z)
    lut_path = args.lut_path  # User-provided LUT file path
    polynomial_degree = args.polynomial_degree
    resample_rate = (
        args.resample_rate
    )  # User-provided resample rate (default: 1 minute)

    # Run the update_lut_force_balance function asynchronously
    asyncio.run(
        update_lut_force_balance(
            force_type,
            start_time,
            end_time,
            axis,
            lut_path,
            polynomial_degree,
            resample_rate,
        )
    )


if __name__ == "__main__":
    run()
