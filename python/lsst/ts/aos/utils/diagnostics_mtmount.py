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

__all__ = ["DiagnosticsMTMount"]

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
from astropy.time.core import Time
from pandas.core.frame import DataFrame

from .diagnostics_default import DiagnosticsDefault
from .enum import EfdName


class DiagnosticsMTMount(DiagnosticsDefault):
    """Main telescope (MT) mount diagnostics class to query and plot the data.

    Parameters
    ----------
    efd_name : enum `EfdName`, optional
        Engineer facility database (EFD) name. (the default is
        EfdName.Summit)
    """

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

        data, time_operation = await self.query_data(
            "MTMount.azimuth",
            [
                "actualPosition",
                "actualTorque",
                "actualVelocity",
                "demandPosition",
                "demandVelocity",
                "private_sndStamp",
            ],
            time_start,
            time_end,
            realign_time,
        )
        return data, time_operation

    async def get_data_elevation(
        self,
        time_start: Time,
        time_end: Time,
        realign_time: bool = True,
    ) -> tuple[DataFrame, numpy.typing.NDArray[np.float64]]:
        """
        Query and return the elevation data.

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
            Elevation data.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        data, time_operation = await self.query_data(
            "MTMount.elevation",
            [
                "actualPosition",
                "actualTorque",
                "actualVelocity",
                "demandPosition",
                "demandVelocity",
                "private_sndStamp",
            ],
            time_start,
            time_end,
            realign_time,
        )
        return data, time_operation

    def plot_velocity(
        self,
        data: DataFrame,
        time_operation: numpy.typing.NDArray[np.float64],
        title: None | str = None,
    ) -> None:
        """Plot the velocity.

        Parameters
        ----------
        data_azimuth : `pandas.core.frame.DataFrame`
            Azimuth data.
        time_operation : `numpy.ndarray`
            Realigned operation time in second (0-based).
        title : None or `str`, optional
            Title. (the default is None)
        """

        plt.figure()
        plt.plot(time_operation, data.actualVelocity)

        plt.xlabel("Time (sec)")
        plt.ylabel("Velocity (deg/sec)")

        if title is not None:
            plt.title(title)

        plt.show()
