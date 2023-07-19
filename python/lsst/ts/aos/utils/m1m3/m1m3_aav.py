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

import numpy as np
import pandas as pd
from astropy.time import Time
from lsst_efd_client import EfdClient


class AccelerationAndVelocities:
    def __init__(self, efd_name: str):
        self.efd_name = efd_name

    async def find_non_zero(
        self, start_time: Time, end_time: Time
    ) -> (pd.DataFrame, pd.DataFrame):
        """Find times when elevations and azimuth MTMount axis moved."""

        async def find_axis(axis: str) -> pd.DataFrame:
            query = (
                "SELECT actualVelocity, demandVelocity "
                f'FROM "efd"."autogen"."lsst.sal.MTMount.{axis}" '
                f"WHERE time > '{start_time.isot}+00:00' AND time < '{end_time.isot}+00:00' "
                "AND ((abs(actualVelocity) > 0.01) or (abs(demandVelocity) > 0.01))"
            )
            ret = await self.client.influx_client.query(query)
            ret["timediff"] = ret.index.to_series().diff()

            # calculate derivatives - acceleration
            den = ret["timediff"].dt.total_seconds()
            ret["demandAcceleration"] = ret["demandVelocity"].diff().div(den, axis=0)
            ret["actualAcceleration"] = ret["actualVelocity"].diff().div(den, axis=0)
            return ret

        elevations = await find_axis("elevation")
        azimuths = await find_axis("azimuth")

        return elevations, azimuths

    async def find_applicable_times(
        self, df_moving: pd.DataFrame, df_notmoving: pd.DataFrame
    ) -> pd.DataFrame:
        """Find start and end times when only one axis moved."""
        ret = []

        start = None
        end = None
        for index, row in df_moving[
            (df_moving.timediff > np.timedelta64(1, "s")) | (df_moving.timediff is None)
        ].iterrows():
            end = index - row["timediff"]
            if start is not None:
                ret.append([start, end, len(df_notmoving[start:end].index) == 0])
            start = index

        return pd.DataFrame(ret, columns=["start", "end", "use"])

    async def fit_aav(
        self, start_time: Time, end_time: Time, extrapolate_baseline: bool
    ) -> None:
        self.client = EfdClient(self.efd_name)
        elevations, azimuths = await self.find_non_zero(start_time, end_time)
        intervals = await self.find_applicable_times(elevations, azimuths)
        print(intervals)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update Acceleration And Velocities tables based on registered hardpoint load cells loads"
    )
    parser.add_argument(
        "start_time",
        type=Time,
        help="Start time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'",
    )
    parser.add_argument(
        "end_time", type=Time, help="End time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'"
    )

    parser.add_argument(
        "--axes",
        choices=["X", "Y", "Z", "XY", "XZ", "YZ", "XZY"],
        default="XYZ",
        help="Axis of the force balance to be updated. Defaults to all axis.",
    )
    parser.add_argument(
        "--efd",
        default="usdf_efd",
        help="EFD name. Defaults to usdf_efd",
    )
    parser.add_argument(
        "--m1m3-config", default=None, help="Path to the LUT file to be updated"
    )

    return parser.parse_args()


def run() -> None:
    args = parse_arguments()

    aav = AccelerationAndVelocities(args.efd)

    asyncio.run(aav.fit_aav(args.start_time, args.end_time, False))


if __name__ == "__main__":
    run()
