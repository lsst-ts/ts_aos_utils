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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta
from lsst_efd_client import EfdClient


class AccelerationAndVelocities:
    def __init__(self, efd_name: str):
        self.efd_name = efd_name
        self.actual_velocity_limit = 0.03
        self.demand_velocity_limit = 1e-5

    async def find_non_zero(
        self, start_time: Time, end_time: Time
    ) -> (pd.DataFrame, pd.DataFrame):
        """Find times when elevations and azimuth MTMount axis moved."""

        async def find_axis(axis: str) -> pd.DataFrame:
            query = (
                "SELECT actualVelocity, demandVelocity "
                f'FROM "efd"."autogen"."lsst.sal.MTMount.{axis}" '
                f"WHERE time > '{start_time.isot}+00:00' AND time < '{end_time.isot}+00:00' "
                f"AND ((abs(actualVelocity) > {self.actual_velocity_limit:f}) or "
                f"(abs(demandVelocity) > {self.demand_velocity_limit:f}))"
            )
            print("Quering EFD:", query)
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
            if start is not None and (end - start) > TimeDelta(0.5, format="sec"):
                ret.append([start, end, len(df_notmoving[start:end].index) == 0])
            start = index

        return pd.DataFrame(ret, columns=["start", "end", "use"])

    async def load_hardpoints(self, start_time: Time, end_time: Time) -> pd.DataFrame:
        print(f"Retrieving HP data for {start_time} - {end_time}..", end="")
        ret = await self.client.select_time_series(
            "lsst.sal.MTM1M3.hardpointActuatorData",
            [f"measuredForce{hp}" for hp in range(6)]
            + [f"f{a}" for a in "xyz"]
            + [f"m{a}" for a in "xyz"],
            Time(start_time),
            Time(end_time),
        )
        print("OK")
        return ret

    async def fit_aav(self, start_time: Time, end_time: Time, plot: bool) -> None:
        self.client = EfdClient(self.efd_name)
        self._plot = plot

        elevations, azimuths = await self.find_non_zero(start_time, end_time)
        intervals = await self.find_applicable_times(elevations, azimuths)
        fit = pd.DataFrame()
        for index, row in intervals[intervals.use].iterrows():
            block_start = row["start"] - pd.Timedelta("2s")
            block_end = row["end"] + 2
            hardpoints = await self.load_hardpoints(block_start, block_end)
            elevations_hardpoints = elevations[
                (elevations.index >= block_start) & (elevations.index <= block_end)
            ]
            fit = pd.concat(
                [
                    fit,
                    pd.merge(
                        elevations_hardpoints,
                        hardpoints,
                        how="outer",
                        left_index=True,
                        right_index=True,
                    ),
                ]
            )
        print("Hardpoint data retrieved")
        print(elevations_hardpoints.describe())
        print(fit)
        fit.drop(columns=["timediff"], inplace=True)
        print(fit.dtypes)
        # fit = fit.interpolate(method="polynomial", order=5)
        fit = fit.interpolate(method="time")  # , order=5)
        print(fit.describe())
        print(fit)
        fit.to_hdf("fit.hd5", "fit")
        plt.plot(fit["demandVelocity"], fit["my"], ".")
        plt.plot(fit["demandAcceleration"], fit["my"], ".")
        plt.show()
        # print(df)


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
    parser.add_argument(
        "--plot", default=False, action="store_true", help="Plot graphs during"
    )

    return parser.parse_args()


async def run_loop() -> None:
    args = parse_arguments()

    aav = AccelerationAndVelocities(args.efd)
    await aav.fit_aav(args.start_time, args.end_time, args.plot)


def run() -> None:
    asyncio.run(run_loop())


if __name__ == "__main__":
    run()
