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

__all__ = ["DiagnosticsDefault"]

import numpy as np
import numpy.typing
from astropy.time.core import Time
from lsst_efd_client import EfdClient
from pandas.core.frame import DataFrame

from .enum import EfdName


class DiagnosticsDefault:
    """Default diagnostics class to query the data.

    Parameters
    ----------
    index : `int` or None, optional
        SAL index. (the default is None).
    efd_name : enum `EfdName`, optional
        Engineer facility database (EFD) name. (the default is
        EfdName.Summit)

    Attributes
    ----------
    efd_client : `lsst_efd_client.efd_helper.EfdClient`
        EFD client.
    """

    def __init__(
        self, index: int | None = None, efd_name: EfdName = EfdName.Summit
    ) -> None:
        self._index = index
        self.efd_client = self._retrieve_efd_client(efd_name)

    def _retrieve_efd_client(self, efd_name: EfdName) -> EfdClient:
        """
        Retrieve a client to engineer facility database (EFD).

        Parameters
        ----------
        efd_name : enum `EfdName`
            EFD name.

        Returns
        -------
        `EfdClient`
            The interface object between the Nublado and summit/Tucson EFD.
        """

        match efd_name:
            case EfdName.Summit:
                name = "summit_efd"

            case EfdName.TucsonTeststand:
                name = "tucson_teststand_efd"

            case EfdName.Base:
                name = "base_efd"

            case EfdName.Usdf:
                name = "usdf_efd"

        return EfdClient(name)

    def get_fields_array(self, components: list[str], numbers: list[int]) -> list[str]:
        """Get the fields of array.

        Parameters
        ----------
        components : `list`
            List of the components.
        numbers : `list`
            List of the number of array elements.

        Returns
        -------
        fields : `list`
            Fields.
        """
        fields: list = list()
        for component, number in zip(components, numbers):
            fields = fields + [f"{component}{idx}" for idx in range(number)]

        return fields

    async def query_data(
        self,
        name: str,
        fields: list[str],
        time_start: Time,
        time_end: Time,
        realign_time: bool,
        name_timestamp: str = "private_sndStamp",
    ) -> tuple[DataFrame, numpy.typing.NDArray[np.float64]]:
        """Query the EFD data.

        Parameters
        ----------
        name : `str`
            Component.topic name (such as "MTHexapod.actuators").
        fields : `list`
            Fields.
        time_start : `astropy.time.core.Time`
            Start time.
        time_end : `astropy.time.core.Time`
            End time.
        realign_time : `bool`
            Realign the timestamp to origin (0-based) or not.
        name_timestamp : `str`, optional
            Name of the timestamp. (the default is "private_sndStamp")

        Returns
        -------
        data : `pandas.core.frame.DataFrame`
            Data.
        time_operation : `numpy.ndarray`
            Operation time.
        """
        data = await self.efd_client.select_time_series(
            f"lsst.sal.{name}",
            fields=fields,
            start=time_start,
            end=time_end,
            index=self._index,
        )

        # Realign the time origin to 0
        if hasattr(data, name_timestamp):
            timestamps = getattr(data, name_timestamp)
            time_operation = (
                np.array(timestamps.subtract(timestamps[0]))
                if realign_time
                else np.array(timestamps)
            )
        else:
            time_operation = np.array([])

        return data, time_operation
