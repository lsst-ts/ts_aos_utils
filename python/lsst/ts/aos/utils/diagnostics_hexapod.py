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

__all__ = ["DiagnosticsHexapod"]

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
from astropy.time.core import Time
from matplotlib.axes import SubplotBase
from pandas.core.frame import DataFrame

from .diagnostics_default import DiagnosticsDefault


class DiagnosticsHexapod(DiagnosticsDefault):
    """Hexapod diagnostics class to query and plot the data.

    Parameters
    ----------
    is_camera_hexapod : `bool`, optional
        This is the camera hexapod or not. If not, it will be the M2 hexapod.
        (the default is True)
    is_summit : `bool`, optional
        This is running on the summit or not. (the default is True)
    """

    # Number of Copley drives to control the actuators
    NUM_DRIVE = 3

    # Number of actuators
    # Drive 0 controls the actuator 1 and 2
    # Drive 1 controls the actuator 3 and 4
    # Drive 2 controls the actuator 5 and 6
    NUM_ACTUATOR = 6

    # Number of the degree of freedom (DOF): (x, y, z, u, v ,w)
    NUM_DOF = 6

    NM_TO_UM = 1e-3

    def __init__(self, is_camera_hexapod: bool = True, is_summit: bool = True) -> None:
        index = 1 if is_camera_hexapod else 2

        super().__init__(index=index, is_summit=is_summit)

    async def get_data_actuators(
        self,
        time_start: Time,
        time_end: Time,
        realign_time: bool = True,
    ) -> tuple[DataFrame, numpy.typing.NDArray[np.float64]]:
        """
        Query and return the actuator data.

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
            Actuator data.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        components = ["calibrated", "raw", "positionError"]
        fields = self.get_fields_array(
            components, [self.NUM_ACTUATOR] * len(components)
        )

        data, time_operation = await self.query_data(
            "MTHexapod.actuators",
            fields + ["private_sndStamp"],
            time_start,
            time_end,
            realign_time,
        )
        return data, time_operation

    async def get_data_application(
        self,
        time_start: Time,
        time_end: Time,
        realign_time: bool = True,
    ) -> tuple[DataFrame, numpy.typing.NDArray[np.float64]]:
        """
        Query and return the application data.

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
            Application data.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        components = ["demand", "position", "error"]
        fields = self.get_fields_array(components, [self.NUM_DOF] * len(components))

        data, time_operation = await self.query_data(
            "MTHexapod.application",
            fields + ["private_sndStamp"],
            time_start,
            time_end,
            realign_time,
        )
        return data, time_operation

    async def get_data_electrical(
        self,
        time_start: Time,
        time_end: Time,
        realign_time: bool = True,
    ) -> tuple[DataFrame, numpy.typing.NDArray[np.float64]]:
        """
        Query and return the electrical data.

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
            Electrical data.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        components = [
            "copleyStatusWordDrive",
            "copleyLatchingFaultStatus",
            "motorCurrent",
            "busVoltage",
        ]
        fields = self.get_fields_array(
            components,
            [self.NUM_ACTUATOR, self.NUM_ACTUATOR, self.NUM_ACTUATOR, self.NUM_DRIVE],
        )

        data, time_operation = await self.query_data(
            "MTHexapod.electrical",
            fields + ["private_sndStamp"],
            time_start,
            time_end,
            realign_time,
        )
        return data, time_operation

    async def get_data_controller_state(
        self, time_start: Time, time_end: Time
    ) -> DataFrame:
        """
        Query and return the controller state data.

        Parameters
        ----------
        time_start : `astropy.time.core.Time`
            Start time.
        time_end : `astropy.time.core.Time`
            End time.

        Returns
        -------
        data : `pandas.core.frame.DataFrame`
            Controller state data.
        """

        data, _ = await self.query_data(
            "MTHexapod.logevent_controllerState",
            [
                "controllerState",
                "enabledSubstate",
                "applicationStatus",
                "private_sndStamp",
            ],
            time_start,
            time_end,
            False,
        )
        return data

    def map_bit_value(
        self,
        data: numpy.typing.NDArray[np.float64],
        num_bits: int,
    ) -> numpy.typing.NDArray[np.int64]:
        """Map the bit value.

        Parameters
        ----------
        data : `numpy.ndarray`
            1D or 2D array data.
        num_bits : `int`
            Number of bits.

        Returns
        -------
        bitmap : `numpy.ndarray`
            Mapped bit values. The first dimension is the bit.

        Raises
        ------
        `ValueError`
            When the input data is not 1D or 2D array.
        """

        dimension = len(data.shape)
        if dimension == 1:
            bitmap = np.zeros((num_bits, data.shape[0]), dtype=int)
        elif dimension == 2:
            bitmap = np.zeros((num_bits, data.shape[0], data.shape[1]), dtype=int)
        else:
            raise ValueError("Input data should be 1D or 2D array only.")

        for bit in range(num_bits):
            value = 2**bit
            bitmap[bit, :] = (data & value) // value

        return bitmap

    def plot_hexapod_position(
        self,
        data_application: DataFrame,
        time_operation: numpy.typing.NDArray[np.float64],
    ) -> None:
        """
        Plot the data of hexapod position.

        Parameters
        -------
        data_application : `pandas.core.frame.DataFrame`
            Application data.
        time_operation : `numpy.ndarray`
            Realigned operation time in second (0-based).
        """

        fig, axs = plt.subplots(2, 3)
        axes_demand = [f"demand{idx}" for idx in range(self.NUM_DOF)]
        axes_position = [f"position{idx}" for idx in range(self.NUM_DOF)]
        axes_title = ["x", "y", "z", "u", "v", "w"]

        for idx in range(self.NUM_DOF):
            row = 0 if idx < 3 else 1
            col = idx % 3

            self._subplot(
                axs[row, col],
                [data_application],
                [time_operation],
                ["r"],
                axes_demand[idx],
                title=axes_title[idx],
            )
            self._subplot(
                axs[row, col],
                [data_application],
                [time_operation],
                ["b"],
                axes_position[idx],
                title=axes_title[idx],
            )

        axs[0, 0].set_ylabel("Position (um)")
        axs[1, 0].set_ylabel("Angle (degree)")
        axs[1, 1].set_xlabel("Time (sec)")

        fig.suptitle("Hexapod Position")
        fig.tight_layout()
        fig.legend(["demand", "position"], loc="lower right")

        plt.show()

    def _subplot(
        self,
        ax: SubplotBase,
        datas: list[DataFrame],
        time_operations: list[numpy.typing.NDArray[np.float64]],
        colors: list[str],
        component: str,
        title: str | None = None,
        scale: float = 1.0,
    ) -> None:
        """Subplot.

        Parameters
        ----------
        ax : `SubplotBase`
            Subplot axes.
        datas : `list`
            Datas.
        time_operations : `list`
            Operation times in second.
        colors : `list`
            Colors.
        component : `str`
            Componenet.
        title : `str` or None, optional
            Title. If None, the component is used. (the default is None)
        scale : `float`, optional
            Scaling of data. (the default is 1.0)
        """
        for data, time_operation, color in zip(datas, time_operations, colors):
            ax.plot(time_operation, getattr(data, component) * scale, color)

        ax_title = component if (title is None) else title
        ax.set_title(ax_title)

    def plot_hexapod_position_error(
        self,
        data_application: DataFrame,
        time_operation: numpy.typing.NDArray[np.float64],
    ) -> None:
        """
        Plot the data of hexapod position error.

        Parameters
        -------
        data_application : `pandas.core.frame.DataFrame`
            Application data.
        time_operation : `numpy.ndarray`
            Realigned operation time in second (0-based).
        """

        fig, axs = plt.subplots(2, 3)
        axes_error = [f"error{idx}" for idx in range(self.NUM_DOF)]
        axes_title = [rf"$\Delta${axis}" for axis in "xyzuvw"]

        for idx in range(self.NUM_DOF):
            row = 0 if idx < 3 else 1
            col = idx % 3

            self._subplot(
                axs[row, col],
                [data_application],
                [time_operation],
                ["b"],
                axes_error[idx],
                title=axes_title[idx],
            )

        axs[0, 0].set_ylabel("Position Error (um)")
        axs[1, 0].set_ylabel("Angle Error (degree)")
        axs[1, 1].set_xlabel("Time (sec)")

        fig.suptitle("Hexapod Position Error (position - demand)")
        fig.tight_layout()

        plt.show()

    def plot_actuator_position(
        self,
        data_actuators: DataFrame,
        time_operation: numpy.typing.NDArray[np.float64],
    ) -> None:
        """
        Plot the data of actuator position.

        Parameters
        -------
        data_actuators : `pandas.core.frame.DataFrame`
            Actuators data.
        time_operation : `numpy.ndarray`
            Realigned operation time in second (0-based).
        """

        fig, axs = plt.subplots(2, 3)

        axes_error = [f"raw{idx}" for idx in range(self.NUM_ACTUATOR)]

        for idx, axis in enumerate(axes_error):
            row = 0 if idx < 3 else 1
            col = idx % 3

            # The unit of raw data is nm.
            self._subplot(
                axs[row, col],
                [data_actuators],
                [time_operation],
                ["b"],
                axis,
                scale=self.NM_TO_UM,
            )

        axs[0, 0].set_ylabel("Position (um)")
        axs[1, 0].set_ylabel("Position (um)")
        axs[1, 1].set_xlabel("Time (sec)")

        fig.suptitle("Encoder Position")
        fig.tight_layout()

        plt.show()

    def plot_actuator_position_error(
        self,
        data_actuators: DataFrame,
        time_operation: numpy.typing.NDArray[np.float64],
    ) -> None:
        """
        Plot the data of actuator position error.

        Parameters
        -------
        data_actuators : `pandas.core.frame.DataFrame`
            Actuators data.
        time_operation : `numpy.ndarray`
            Realigned operation time in second (0-based).
        """

        fig, axs = plt.subplots(2, 3)
        axes_error = [f"positionError{idx}" for idx in range(self.NUM_ACTUATOR)]

        for idx, axis in enumerate(axes_error):
            row = 0 if idx < 3 else 1
            col = idx % 3

            self._subplot(
                axs[row, col],
                [data_actuators],
                [time_operation],
                ["b"],
                axis,
            )

        axs[0, 0].set_ylabel("Position Error (um)")
        axs[1, 0].set_ylabel("Position Error (um)")
        axs[1, 1].set_xlabel("Time (sec)")

        fig.suptitle("Actuator Position Error (actual - demand)")
        fig.tight_layout()

        plt.show()

    def plot_actuator_position_velocity(
        self,
        data_actuators: DataFrame,
        time_operation: numpy.typing.NDArray[np.float64],
    ) -> None:
        """
        Plot the data of actuator position and velocity.

        Parameters
        -------
        data_actuators : `pandas.core.frame.DataFrame`
            Actuators data.
        time_operation : `numpy.ndarray`
            Realigned operation time in second (0-based).
        """

        fig, axs = plt.subplots(1, 2)

        time_diff = np.diff(time_operation)

        for idx in range(self.NUM_ACTUATOR):
            position = getattr(data_actuators, f"raw{idx}") * self.NM_TO_UM
            axs[0].plot(time_operation, position)
            axs[1].plot(time_operation[:-1], np.diff(position) / time_diff)

        for ax in axs:
            ax.legend([f"actuator{idx}" for idx in range(self.NUM_ACTUATOR)])
            ax.set_xlabel("Time (sec)")

        axs[0].set_ylabel("Position (um)")
        axs[1].set_ylabel("Velocity (um/sec)")

        fig.suptitle("Encoder Position and Velocity")
        fig.tight_layout()

        plt.show()

    def plot_actuator_current(
        self,
        data_electrical: DataFrame,
        time_operation: numpy.typing.NDArray[np.float64],
    ) -> None:
        """
        Plot the data of actuator current.

        Parameters
        -------
        data_electrical : `pandas.core.frame.DataFrame`
            Electrical data.
        time_operation : `numpy.ndarray`
            Realigned operation time in second (0-based).
        """

        fig, axs = plt.subplots(2, 3)

        axes_current = [f"motorCurrent{idx}" for idx in range(self.NUM_ACTUATOR)]

        for idx, axis in enumerate(axes_current):
            row = 0 if idx < 3 else 1
            col = idx % 3

            self._subplot(
                axs[row, col],
                [data_electrical],
                [time_operation],
                ["b"],
                axis,
            )

        axs[0, 0].set_ylabel("Current (A)")
        axs[1, 0].set_ylabel("Current (A)")
        axs[1, 1].set_xlabel("Time (sec)")

        fig.suptitle("Actuator Current")
        fig.tight_layout()

        plt.show()

    def plot_bus_voltage(
        self,
        data_electrical: DataFrame,
        time_operation: numpy.typing.NDArray[np.float64],
    ) -> None:
        """
        Plot the data of bus voltage.

        Parameters
        -------
        data_electrical : `pandas.core.frame.DataFrame`
            Electrical data.
        time_operation : `numpy.ndarray`
            Realigned operation time in second (0-based).
        """

        fig, axs = plt.subplots(1, 3)

        axes_voltage = [f"busVoltage{idx}" for idx in range(self.NUM_DRIVE)]

        for idx, axis in enumerate(axes_voltage):
            self._subplot(
                axs[idx],
                [data_electrical],
                [time_operation],
                ["b"],
                axis,
            )

        axs[0].set_ylabel("Voltage (V)")
        axs[1].set_xlabel("Time (sec)")

        fig.suptitle("Bus Voltage")
        fig.tight_layout()

        plt.show()

    def plot_actuator_bit(
        self,
        time_operation: numpy.typing.NDArray[np.int64],
        bitmap: numpy.typing.NDArray[np.int64],
        bit: int,
        title: str,
    ) -> None:
        """Plot the bit of 6 actuators.

        Parameters
        ----------
        time_operation : `numpy.ndarray`
            Operation time in seconds.
        bitmap : `numpy.ndarray`
            Bitmap. This should be a 3D array. The first dimension is the bit.
            The second dimension is the actuator, which has 6. The third
            dimension is the bit value with time.
        bit : `int`
            Bit.
        title : `str`
            title.
        """

        fig, axs = plt.subplots(2, 3)

        for idx in range(self.NUM_ACTUATOR):
            row = 0 if idx < 3 else 1
            col = idx % 3

            axs[row, col].plot(time_operation, bitmap[bit, idx, :])
            axs[row, col].set_title(f"actuator{idx}")

        axs[0, 0].set_ylabel("Value (On/Off)")
        axs[1, 0].set_ylabel("Value (On/Off)")
        axs[1, 1].set_xlabel("Time (sec)")

        fig.suptitle(title)
        fig.tight_layout()

        plt.show()

    def plot_application_status(
        self,
        data_controller_state: DataFrame,
        bit: int,
        timestamp_origin: float | None = None,
        num_bit: int = 15,
        title: str | None = None,
    ) -> None:
        """Plot the application status.

        Parameters
        ----------
        data_controller_state : `DataFrame`
            Data of the controller state.
        bit : `int`
            Bit.
        timestamp_origin : `float` or None, optional
            Origin of the timestamp. If not None, this will be the origin of
            the x-axis. (the default is None)
        num_bit : `int`, optional
            Number of bit. (the default is 15)
        title : `str` or None, optional
            Title. (the default is None)
        """

        bitmap = self.map_bit_value(data_controller_state.applicationStatus, num_bit)

        plt.figure()

        time = (
            data_controller_state.private_sndStamp
            if (timestamp_origin is None)
            else (data_controller_state.private_sndStamp - timestamp_origin)
        )

        plt.plot(time, bitmap[bit, :], "bx")
        plt.xlabel("Time (sec)")
        plt.ylabel("Value (On/Off)")

        if title is not None:
            plt.title(title)

        plt.show()
