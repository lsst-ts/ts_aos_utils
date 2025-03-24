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

__all__ = ["DiagnosticsMTDome"]

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
from astropy.time.core import Time
from pandas.core.frame import DataFrame

from .diagnostics_default import DiagnosticsDefault
from .enum import EfdName


class DiagnosticsMTDome(DiagnosticsDefault):
    """Main telescope (MT) dome diagnostics class to query and plot the data.

    Parameters
    ----------
    efd_name : enum `EfdName`, optional
        Engineer facility database (EFD) name. (the default is
        EfdName.Summit)
    """

    NUM_DRIVE_AZIMUTH = 5

    def __init__(self, efd_name: EfdName = EfdName.Summit) -> None:
        super().__init__(efd_name=efd_name)

    async def get_data_azimuth(
        self,
        time_start: Time,
        time_end: Time,
        realign_time: bool = True,
    ) -> tuple[DataFrame, numpy.typing.NDArray[np.float64]]:
        """
        Query and return the azimuth data.

        Parameters
        ----------
        time_start : `astropy.time.core.Time`
            Start time.
        time_end : `astropy.time.core.Time`
            End time.
        realign_time : `bool`, optional
            Realign the timestamp to origin or not (0-based). (the default is
            True)

        Returns
        -------
        data : `pandas.core.frame.DataFrame`
            Azimuth data.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        components = ["driveTorqueActual", "driveTorqueCommanded", "driveCurrentActual"]
        fields = self.get_fields_array(
            components, [self.NUM_DRIVE_AZIMUTH] * len(components)
        )

        data, time_operation = await self.query_data(
            "MTDome.azimuth",
            fields
            + [
                "positionActual",
                "positionCommanded",
                "velocityActual",
                "velocityCommanded",
                "private_sndStamp",
            ],
            time_start,
            time_end,
            realign_time,
        )
        return data, time_operation

    def calculate_target_position(
        self,
        data: DataFrame,
        time_operation: numpy.typing.NDArray[np.float64],
    ) -> numpy.typing.NDArray[np.float64]:
        """Calculate the target position.

        Parameters
        ----------
        data : `pandas.core.frame.DataFrame`
            Azimuth data.
        time_operation : `numpy.ndarray`
            Realigned operation time in second (0-based).


        Returns
        -------
        target_position_final : `numpy.ndarray`
            Target position.
        """

        target_position = np.array(data.positionCommanded)
        nan_indices = np.where(np.isnan(target_position))[0]

        if len(nan_indices) == 0:
            return target_position

        if nan_indices[0] == 0:
            data_to_fill = target_position[nan_indices[-1] + 1]
        else:
            data_to_fill = target_position[0]

        target_position[nan_indices] = data_to_fill

        delta_time = np.diff(time_operation, prepend=0.0)
        delta_target_position = np.array(data.velocityCommanded) * delta_time
        delta_target_position_cumsum = np.cumsum(delta_target_position)

        target_position_final = target_position + delta_target_position_cumsum
        for idx in range(len(target_position_final)):
            if target_position_final[idx] >= 360.0:
                target_position_final[idx] -= 360.0

        return target_position_final

    def plot_azimuth_position_velocity(
        self,
        data: DataFrame,
        time_operation: numpy.typing.NDArray[np.float64],
    ) -> None:
        """Plot the azimuth position and velocity.

        Parameters
        ----------
        data : `pandas.core.frame.DataFrame`
            Azimuth data.
        time_operation : `numpy.ndarray`
            Realigned operation time in second (0-based).
        """

        target_position = self.calculate_target_position(data, time_operation)

        fig, axs = plt.subplots(1, 2)

        # Plot the position data
        axs[0].plot(time_operation, target_position)
        axs[0].plot(time_operation, data.positionActual)

        axs[0].legend(["Target", "Actual"])
        axs[0].set_xlabel("Time (sec)")
        axs[0].set_ylabel("Position (deg)")
        axs[0].set_title("Position")

        # Plot the velocity data
        axs[1].plot(time_operation, data.velocityCommanded)
        axs[1].plot(time_operation, data.velocityActual)

        axs[1].legend(["Commanded", "Actual"])
        axs[1].set_xlabel("Time (sec)")
        axs[1].set_ylabel("Velocity (deg/sec)")
        axs[1].set_title("Velocity")

        fig.tight_layout()

        plt.show()

    def plot_azimuth_drive_current(
        self,
        data: DataFrame,
        time_operation: numpy.typing.NDArray[np.float64],
    ) -> None:
        """Plot the azimuth drive current.

        Parameters
        ----------
        data : `pandas.core.frame.DataFrame`
            Azimuth data.
        time_operation : `numpy.ndarray`
            Realigned operation time in second (0-based).
        """

        plt.figure()
        for idx in range(self.NUM_DRIVE_AZIMUTH):
            plt.plot(time_operation, getattr(data, f"driveCurrentActual{idx}"))

        plt.legend([f"drive {idx}" for idx in range(self.NUM_DRIVE_AZIMUTH)])
        plt.xlabel("Time (sec)")
        plt.ylabel("Current (A)")

        plt.title("Drive Current")

        plt.show()
