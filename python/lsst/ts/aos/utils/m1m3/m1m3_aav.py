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

__all__ = ["AccelerationAndVelocities"]

import argparse
import asyncio
import os
import pathlib

import numpy as np

# import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from lsst.ts.criopy import M1M3FATable
from lsst.ts.criopy.m1m3 import AccelerationAndVelocitiesFitter, ForceCalculator
from lsst.ts.idl.enums.MTM1M3 import DetailedState
from lsst_efd_client import EfdClient
from tqdm import tqdm

tqdm.pandas()


class AccelerationAndVelocities:
    """Compute fir of M1M3 acceleration and velocities.

    Attributes
    ----------
    azimuths : pd.DataFrame
        Mount azimuth. Contains actual and demand Position, Velocity and Acceleration.
    elevations : pd.DataFrame
        Mount elevation. Contains actual and demand Position, Velocity and Acceleration.
    interpolated : pd.DataFrame
        Interpolated raw data.
    intervals : pd.DataFrame
        Contains start, end and use flag (whenever to use the interval) of
        intervals suitable for fitting.
    mirror : pd.DataFrame
        Mirror forces. Contains interpolated data, with extra X0..11, Y0..99 and Z0..155 columns - those
        contain data derived from hardpoint forces (f[xyz], m[xyz]). Those are
        the per-mirror (XYZ) forces the algorithm needs to counter-act.
    raw : pd.DataFrame
        Raw data. Concatenation of azimuths, elevations and hardpoint forces.
        Contains null/NaNs, as the three tables are merged together.
    """

    def __init__(self, efd_name: str, config: str):
        self.efd_name = efd_name
        self.force_calculator = ForceCalculator(config)

        self.actual_velocity_limit = 0.01
        self.demand_velocity_limit = 0.01
        self.detailed_states: pd.DataFrame | None = None

        self.azimuths: pd.DataFrame | None = None
        self.elevations: pd.DataFrame | None = None
        self.interpolated: pd.DataFrame | None = None
        self.intervals: pd.DataFrame | None = None
        self.mirror: pd.DataFrame | None = None
        self.raw: pd.DataFrame | None = None

    async def load_detailed_state(
        self, start_time: Time, end_time: Time
    ) -> pd.DataFrame:
        # find last state before start_time
        print(f"Retrieve last detailedState before {start_time.isot}..", end="")
        query = (
            "SELECT detailedState "
            'FROM "efd"."autogen"."lsst.sal.MTM1M3.logevent_detailedState" '
            f"WHERE time <= '{start_time.isot}+00:00' ORDER BY time DESC LIMIT 1"
        )
        ret = await self.client.influx_client.query(query)
        detailed_states = await self.client.select_time_series(
            "lsst.sal.MTM1M3.logevent_detailedState",
            "detailedState",
            start_time,
            end_time,
        )
        self.detailed_states = pd.concat([ret, detailed_states])
        self.detailed_states.set_index(
            pd.DatetimeIndex(self.detailed_states.index), inplace=True
        )

    def was_raised(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        last_state = self.detailed_states[
            self.detailed_states.index < pd.to_datetime(start, utc=True)
        ]["detailedState"].iloc[-1]
        if DetailedState(last_state) not in (
            DetailedState.ACTIVE,
            DetailedState.ACTIVEENGINEERING,
        ):
            return False
        states = self.detailed_states[
            (Time(self.detailed_states.index) >= start)
            & (Time(self.detailed_states.index) <= end)
        ]
        return (
            len(
                states[
                    ~states.detailedState.isin(
                        [
                            DetailedState.ACTIVE,
                            DetailedState.ACTIVEENGINEERING,
                        ]
                    )
                ].index
            )
            == 0
        )

    async def find_az_el(self, start_time: Time, end_time: Time) -> None:
        """Find times when elevations and azimuth MTMount axis moved. Store
        elevations and azimuth to elevations and azimuths attributes."""

        async def find_axis(axis: str) -> pd.DataFrame:
            query = (
                "SELECT timestamp, demandPosition, actualPosition, actualVelocity, demandVelocity "
                f'FROM "efd"."autogen"."lsst.sal.MTMount.{axis}" '
                f"WHERE time > '{start_time.isot}+00:00' AND time < '{end_time.isot}+00:00'"
            )

            # query_non_zero = (
            #    query
            # + f" AND ((abs(actualVelocity) > {self.actual_velocity_limit:f})
            # or "
            #   f"(abs(demandVelocity) > {self.demand_velocity_limit:f}))"
            # )
            print("Querying EFD:", query)
            ret = await self.client.influx_client.query(query)
            # empty response - probably an interval with 0 velocities. Get it
            # out anyway
            if isinstance(ret, dict):
                print("Empty response, quering now", query)
                ret = await self.client.influx_client.query(query)
                if isinstance(ret, dict):
                    raise RuntimeError(
                        f"Cannot retrieve data for MTMount {axis} axis from "
                        f"{start_time.isot} to {end_time.isot}."
                    )

            ret.set_index(
                pd.DatetimeIndex(
                    Time(Time(ret["timestamp"], format="unix_tai"), scale="utc").isot
                ),
                inplace=True,
            )
            ret["timediff"] = ret["timestamp"].diff()

            # calculate derivatives - acceleration
            den = ret["timediff"]
            ret["demandAcceleration"] = ret["demandVelocity"].diff().div(den, axis=0)
            ret["actualAcceleration"] = ret["actualVelocity"].diff().div(den, axis=0)
            return ret

        self.elevations = await find_axis("elevation")
        self.azimuths = await find_axis("azimuth")

    async def find_applicable_times(self) -> None:
        """Find start and end times of any axis movents."""
        ret = []

        movements = pd.concat(
            [
                self.elevations[
                    (abs(self.elevations.actualVelocity) > self.actual_velocity_limit)
                    | (abs(self.elevations.demandVelocity) > self.demand_velocity_limit)
                ],
                self.azimuths[
                    (abs(self.azimuths.actualVelocity) > self.actual_velocity_limit)
                    | (abs(self.azimuths.demandVelocity) > self.demand_velocity_limit)
                ],
            ]
        )

        movements.sort_index(inplace=True)
        movements["timediff"] = movements.index.to_series().diff()

        start = None
        end = None
        for index, row in movements[
            (movements.timediff > np.timedelta64(1, "s")) | (movements.timediff is None)
        ].iterrows():
            end = index - row["timediff"]
            if start is not None and (end - start) > pd.Timedelta(seconds=0.5):
                if self.was_raised(start, end):
                    ret.append(
                        [
                            start,
                            end,
                            True,
                        ]
                        #    if df_notmoving is None
                        #    else len(df_notmoving[start:end].index) == 0,
                    )
                else:
                    print(
                        f"Interval {start.isot} - {end.isot} "
                        "mirror was not raised, ignoring."
                    )
            start = index

        if len(ret) == 0:
            ret = [
                [
                    movements.index[0],
                    movements.index[-1],
                    True,
                ]
            ]

        self.intervals = pd.DataFrame(ret, columns=["start", "end", "use"])
        self.intervals.index = self.intervals["start"]

    def calculate_forces(
        self, fitter: AccelerationAndVelocitiesFitter, coefficients: pd.DataFrame
    ) -> pd.DataFrame:
        results = pd.DataFrame(fitter.aav.values @ coefficients.values) / 1000.0

        def vect_to_fam(row: pd.Series) -> pd.Series:
            applied = self.force_calculator.get_applied_forces_from_series(row)

            return pd.Series(
                [
                    applied.fx,
                    applied.fy,
                    applied.fz,
                    applied.mx,
                    applied.my,
                    applied.mz,
                ],
                index=[f"calculated_f{a}" for a in "xyz"]
                + [f"calculated_m{a}" for a in "xyz"],
            )

        print("Calculating calculated forces - backpropagation")
        calculated = results.progress_apply(vect_to_fam, axis=1)
        calculated.set_index(fitter.aav.index, inplace=True)
        return calculated

    async def load_hardpoints(
        self, start_time: pd.Timestamp, end_time: pd.Timestamp
    ) -> None | pd.DataFrame:
        start = Time(start_time)
        end = Time(end_time)
        print(f"Retrieving HP data for {start.isot} - {end.isot}..", end="")
        ret = await self.client.select_time_series(
            "lsst.sal.MTM1M3.hardpointActuatorData",
            ["timestamp"]
            + [f"measuredForce{hp}" for hp in range(6)]
            + [f"f{a}" for a in "xyz"]
            + [f"m{a}" for a in "xyz"],
            start,
            end,
        )
        if ret.empty:
            print("empty, ignored")
            return None
        ret.set_index(
            pd.DatetimeIndex(
                Time(Time(ret["timestamp"], format="unix_tai"), scale="utc").isot
            ),
            inplace=True,
        )
        print("OK")
        return ret

    def mirror_forces(self, fit: pd.DataFrame) -> pd.DataFrame:
        fa_names = (
            [f"X{x}" for x in range(M1M3FATable.FATABLE_XFA)]
            + [f"Y{y}" for y in range(M1M3FATable.FATABLE_YFA)]
            + [f"Z{z}" for z in range(M1M3FATable.FATABLE_ZFA)]
        )

        def fa_forces(row: pd.Series) -> pd.Series:
            forces = self.force_calculator.forces_and_moments_forces(
                [row["fx"], row["fy"], row["fz"], row["mx"], row["my"], row["mz"]]
            )
            return pd.Series(
                np.concatenate((forces.xForces, forces.yForces, forces.zForces)),
                index=fa_names,
            )

        print("Calculating mirror forces")
        applied = fit.progress_apply(fa_forces, axis=1)
        applied.set_index(fit.index, inplace=True)
        print(applied)

        return fit.merge(applied, how="left", left_index=True, right_index=True)

    async def collect_hardpoint_data(self) -> None:
        assert self.azimuths is not None
        assert self.elevations is not None

        self.raw = pd.DataFrame()
        for index, row in self.intervals[self.intervals.use].iterrows():
            block_start = row["start"]
            block_end = row["end"]
            hardpoints = await self.load_hardpoints(block_start, block_end)
            if hardpoints is None:
                continue
            elevations_hardpoints = self.elevations[
                (self.elevations.index >= block_start)
                & (self.elevations.index <= block_end)
            ]
            azimuths_hardpoints = self.azimuths[
                (self.azimuths.index >= block_start)
                & (self.azimuths.index <= block_end)
            ]
            data = hardpoints.join(
                [azimuths_hardpoints, elevations_hardpoints], how="outer", sort=True
            )

            self.raw = pd.concat([self.raw, data])

    async def fit_aav(
        self,
        start_time: Time,
        end_time: Time,
        set_new: bool,
        plot: bool,
        hd5_debug: None | str,
    ) -> None:
        self.client = EfdClient(self.efd_name)
        self._plot = plot

        await self.load_detailed_state(start_time, end_time)

        await self.find_az_el(start_time, end_time)
        await self.find_applicable_times()

        assert self.intervals is not None
        print(self.intervals)

        assert self.azimuths is not None
        assert self.elevations is not None

        self.elevations.rename(columns=lambda n: f"elevation_{n}", inplace=True)
        self.azimuths.rename(columns=lambda n: f"azimuth_{n}", inplace=True)

        await self.collect_hardpoint_data()
        print("Hardpoint data retrieved")

        assert self.raw is not None

        print(self.raw.describe())
        print(self.raw["elevation_demandPosition"].describe())
        print(self.raw["elevation_actualPosition"].describe())
        print(self.raw["elevation_demandVelocity"].describe())
        print(self.raw["elevation_actualVelocity"].describe())
        print(self.raw["elevation_demandAcceleration"].describe())
        print(self.raw["elevation_actualAcceleration"].describe())

        print(self.raw["azimuth_demandPosition"].describe())
        print(self.raw["azimuth_actualPosition"].describe())
        print(self.raw["azimuth_demandVelocity"].describe())
        print(self.raw["azimuth_actualVelocity"].describe())
        print(self.raw["azimuth_demandAcceleration"].describe())
        print(self.raw["azimuth_actualAcceleration"].describe())

        print(self.raw)
        self.raw.sort_index(inplace=True)
        if hd5_debug is not None:
            self.raw.to_hdf(hd5_debug, "raw")
        self.interpolated = self.raw.interpolate(method="time")
        self.interpolated = self.interpolated[
            self.interpolated.index.to_series().diff() < np.timedelta64(1, "s")
        ]
        print(self.interpolated.describe())
        self.mirror = self.mirror_forces(self.interpolated)
        if hd5_debug is not None:
            self.mirror.to_hdf(hd5_debug, "mirror")
        print(self.mirror)

        # prepare for fit A @ x = B
        self.fitter = AccelerationAndVelocitiesFitter(self.mirror, "actual")
        if hd5_debug is not None:
            self.fitter.aav.to_hdf(hd5_debug, "A")

        self.coefficients, self.fit_residuals = self.fitter.do_fit(self.mirror)
        print("coefficients")
        print(self.coefficients)
        print(self.coefficients.describe())

        if hd5_debug is not None:
            self.coefficients.to_hdf(hd5_debug, "coefficients")

        if set_new:
            self.force_calculator.set_acceleration_and_velocity(self.coefficients)
        else:
            self.force_calculator.update_acceleration_and_velocity(self.coefficients)

        self.calculated = self.calculate_forces(self.fitter, self.coefficients)
        self.mirror_res = pd.DataFrame(
            {
                "fx": self.mirror["fx"] + self.calculated["calculated_fx"],
                "fy": self.mirror["fy"] + self.calculated["calculated_fy"],
                "fz": self.mirror["fz"] + self.calculated["calculated_fz"],
                "mx": self.mirror["mx"] + self.calculated["calculated_mx"],
                "my": self.mirror["my"] + self.calculated["calculated_my"],
                "mz": self.mirror["mz"] + self.calculated["calculated_mz"],
            }
        )
        self.mirror_res.rename(columns=lambda n: f"residuals_{n}", inplace=True)

        self.residuals = self.mirror.join([self.calculated, self.mirror_res])

        if hd5_debug is not None:
            self.residuals.to_hdf(hd5_debug, "residuals")

        save_to = pathlib.Path("new")
        try:
            os.makedirs(save_to)
        except FileExistsError:
            pass

        self.force_calculator.save(pathlib.Path("new"))

        # plt.plot(fit["demandVelocity"], fit["my"], ".")
        # plt.plot(fit["demandAcceleration"], fit["my"], ".")
        # plt.show()
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
    parser.add_argument(
        "--config",
        default=pathlib.Path(os.environ["TS_CONFIG_MTTCS_DIR"]) / "MTM1M3" / "v1",
        help="M1M3 configuration directory",
    )
    parser.add_argument(
        "--set-new",
        default=False,
        action="store_true",
        help="Don't add to existing forces, assuming acceleration and velocity "
        "forces were disabled when acquiring data",
    )
    parser.add_argument(
        "--hd5-debug",
        type=str,
        default=None,
        help="save debug outputs as provided HDF5 file.",
    )

    return parser.parse_args()


async def run_loop() -> None:
    args = parse_arguments()

    aav = AccelerationAndVelocities(args.efd, args.config)
    await aav.fit_aav(
        args.start_time,
        args.end_time,
        args.set_new,
        args.plot,
        args.hd5_debug,
    )


def run() -> None:
    asyncio.run(run_loop())


if __name__ == "__main__":
    run()
