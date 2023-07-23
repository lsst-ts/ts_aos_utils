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
import os
import pathlib

import numpy as np

# import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from lsst.ts.criopy import M1M3FATable
from lsst.ts.criopy.m1m3 import ForceCalculator
from lsst_efd_client import EfdClient
from tqdm import tqdm

# from numpy.linalg import lstsq




class AccelerationAndVelocities:
    def __init__(self, efd_name: str, config: str):
        self.efd_name = efd_name
        self.force_calculator = ForceCalculator(config)

        self.actual_velocity_limit = 0.01
        self.demand_velocity_limit = 1e-5

    async def find_non_zero(
        self, start_time: Time, end_time: Time
    ) -> (pd.DataFrame, pd.DataFrame):
        """Find times when elevations and azimuth MTMount axis moved."""

        async def find_axis(axis: str) -> pd.DataFrame:
            query = (
                "SELECT timestamp, demandPosition, actualPosition, actualVelocity, demandVelocity "
                f'FROM "efd"."autogen"."lsst.sal.MTMount.{axis}" '
                f"WHERE time > '{start_time.isot}+00:00' AND time < '{end_time.isot}+00:00' "
                f"AND ((abs(actualVelocity) > {self.actual_velocity_limit:f}) or "
                f"(abs(demandVelocity) > {self.demand_velocity_limit:f}))"
            )
            print("Querying EFD:", query)
            ret = await self.client.influx_client.query(query)
            ret.set_index("timestamp", inplace=True)
            ret["timediff"] = ret.index.to_series().diff()

            # calculate derivatives - acceleration
            den = ret["timediff"]
            ret["demandAcceleration"] = ret["demandVelocity"].diff().div(den, axis=0)
            ret["actualAcceleration"] = ret["actualVelocity"].diff().div(den, axis=0)
            return ret

        elevations = await find_axis("elevation")
        azimuths = await find_axis("azimuth")

        return elevations, azimuths

    async def find_applicable_times(
        self, df_moving: pd.DataFrame, df_notmoving: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Find start and end times when only one axis moved."""
        ret = []

        start = None
        end = None
        for index, row in df_moving[
            (df_moving.timediff > 1) | (df_moving.timediff is None)
        ].iterrows():
            end = index - row["timediff"]
            if start is not None and (end - start) > 0.5:
                ret.append(
                    [
                        start,
                        end,
                        True
                        if df_notmoving is None
                        else len(df_notmoving[start:end].index) == 0,
                    ]
                )
            start = index

        return pd.DataFrame(ret, columns=["start", "end", "use"])

    async def load_hardpoints(
        self, start_time: Time, end_time: Time
    ) -> None | pd.DataFrame:
        start = Time(start_time, format="unix_tai")
        end = Time(end_time, format="unix_tai")
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
        ret.set_index("timestamp", inplace=True)
        print("OK")
        return ret

    def mirror_forces(self, fit: pd.DataFrame) -> pd.DataFrame:
        forces = {}
        for x in range(M1M3FATable.FATABLE_XFA):
            forces[f"X{x}"] = []
        for y in range(M1M3FATable.FATABLE_YFA):
            forces[f"Y{y}"] = []
        for z in range(M1M3FATable.FATABLE_ZFA):
            forces[f"Z{z}"] = []

        for index, row in tqdm(
            fit.iterrows(), total=fit.shape[0], desc="Calculating mirror forces"
        ):
            mirror_forces = self.force_calculator.forces_and_moments_forces(
                [row["fx"], row["fy"], row["fz"], row["mx"], row["my"], row["mz"]]
            )
            for x in range(M1M3FATable.FATABLE_XFA):
                forces[f"X{x}"].append(mirror_forces.xForces[x])
            for y in range(M1M3FATable.FATABLE_YFA):
                forces[f"Y{y}"].append(mirror_forces.yForces[y])
            for z in range(M1M3FATable.FATABLE_ZFA):
                forces[f"Z{z}"].append(mirror_forces.zForces[z])

        df = pd.DataFrame(forces, index=fit.index)
        print(df)

        return fit.merge(df, how="left", left_index=True, right_index=True)

    async def fit_aav(self, start_time: Time, end_time: Time, plot: bool) -> None:
        self.client = EfdClient(self.efd_name)
        self._plot = plot

        elevations, azimuths = await self.find_non_zero(start_time, end_time)
        intervals = await self.find_applicable_times(elevations, None)  # azimuths)

        elevations.rename(columns=lambda n: f"elevation_{n}", inplace=True)
        azimuths.rename(columns=lambda n: f"azimuth_{n}", inplace=True)

        fit = pd.DataFrame()
        for index, row in intervals[intervals.use].iterrows():
            block_start = row["start"]
            block_end = row["end"]
            hardpoints = await self.load_hardpoints(block_start, block_end)
            if hardpoints is None:
                continue
            elevations_hardpoints = elevations[
                (elevations.index >= block_start) & (elevations.index <= block_end)
            ]

            azimuths_hardpoints = azimuths[
                (azimuths.index >= block_start) & (azimuths.index <= block_end)
            ]
            data = hardpoints.join(
                [azimuths_hardpoints, elevations_hardpoints], how="outer", sort=True
            )

            fit = pd.concat([fit, data])

        fit.iloc[0].fillna(0, inplace=True)
        fit.iloc[-1].fillna(0, inplace=True)

        print("Hardpoint data retrieved")
        print(elevations_hardpoints.describe())
        print(fit)
        fit.to_hdf("raw.hd5", "raw")
        fit = fit.interpolate(method="index")  # polynomial", order=5)
        fit = fit[fit.index.to_series().diff() < 1]
        print(fit.describe())
        print(fit)
        mirror = self.mirror_forces(fit)
        mirror.to_hdf("fit.hd5", "fit")
        print(mirror)

        # prepare for fit A @ x = B
        pos = "demand"  # or "actual"
        el_sin = np.sin(np.radians(fit[f"elevation_{pos}Position"]))
        el_cos = np.cos(np.radians(fit[f"elevation_{pos}Position"]))

        V_x = fit[f"elevation_{pos}Velocity"]
        V_y = fit[f"azimuth_{pos}Velocity"].mul(el_cos)
        V_z = fit[f"azimuth_{pos}Velocity"].mul(el_sin)

        A_x = fit[f"elevation_{pos}Acceleration"]
        A_y = fit[f"azimuth_{pos}Acceleration"].mul(el_cos)
        A_z = fit[f"azimuth_{pos}Acceleration"].mul(el_sin)

        A = pd.DataFrame(
            {
                "V_x2": V_x.pow(2),
                "V_y2": V_y.pow(2),
                "V_z2": V_z.pow(2),
                "V_xz": V_x.mul(V_z),
                "V_yz": V_y.mul(V_z),
                "A_x": A_x,
                "A_y": A_y,
                "A_z": A_z,
            }
        )

        A.fillna(0, inplace=True)

        A.to_hdf("A.hd5", "A")

        ret = []
        res = []

        for fa in tqdm(
            [f"X{x}" for x in range(M1M3FATable.FATABLE_XFA)]
            + [f"Y{y}" for y in range(M1M3FATable.FATABLE_YFA)]
            + [f"Z{z}" for z in range(M1M3FATable.FATABLE_ZFA)],
            desc="Fitting FAs",
        ):
            B = mirror[fa] * -1000.0
            x, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
            ret.append(x)
            res.append(residuals)

        print(ret)

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

    return parser.parse_args()


async def run_loop() -> None:
    args = parse_arguments()

    aav = AccelerationAndVelocities(args.efd, args.config)
    await aav.fit_aav(args.start_time, args.end_time, args.plot)


def run() -> None:
    asyncio.run(run_loop())


if __name__ == "__main__":
    run()
