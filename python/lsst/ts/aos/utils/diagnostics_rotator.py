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

__all__ = ["DiagnosticsRotator"]

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
import pandas as pd
from astropy.time.core import Time
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.fft import fft, fftfreq

from .diagnostics_default import DiagnosticsDefault
from .enum import EfdName


class DiagnosticsRotator(DiagnosticsDefault):
    """Rotator diagnostics class to query and plot the data.

    Parameters
    ----------
    efd_name : enum `EfdName`, optional
        Engineer facility database (EFD) name. (the default is
        EfdName.Summit)
    """

    def __init__(self, efd_name: EfdName = EfdName.Summit) -> None:
        super().__init__(efd_name=efd_name)

    async def get_command_track(
        self,
        time_start: Time,
        time_end: Time,
        realign_time: bool = True,
    ) -> tuple[DataFrame, numpy.typing.NDArray[np.float64]]:
        """
        Get the track command.

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
            Data of the track command.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        # Get the data from EFD
        data, time_operation = await self.query_data(
            "MTRotator.command_track",
            [
                "angle",
                "velocity",
                "tai",
                "private_sndStamp",
            ],
            time_start,
            time_end,
            realign_time,
        )

        # Add the column of time difference in seconds
        self._add_column_time_differece(data, time_start)

        return data, time_operation

    def _add_column_time_differece(
        self, data_frame: DataFrame, time_start: Time
    ) -> None:
        """
        Add the column of time difference in seconds.

        Parameters
        ----------
        data_frame : `pandas.core.frame.DataFrame`
            Data frame.
        time_start : `astropy.time.core.Time`
            Start time.
        """
        if len(data_frame.values) != 0:
            timestamp_start = pd.Timestamp(time_start.value, tz="utc")
            diff_time = data_frame.index - timestamp_start
            diff_time_sec = [
                element.seconds + element.microseconds * 1e-6 for element in diff_time
            ]
            data_frame["time_difference"] = diff_time_sec

    async def get_event_data(
        self, time_start: Time, time_end: Time
    ) -> tuple[DataFrame, DataFrame, DataFrame]:
        """
        Get the event data.

        Parameters
        ----------
        time_start : `astropy.time.core.Time`
            Start time.
        time_end : `astropy.time.core.Time`
            End time.

        Returns
        -------
        data_controller_state : `pandas.core.frame.DataFrame`
            Data of the controller state event.
        data_in_position : `pandas.core.frame.DataFrame`
            Data of the inPosition event.
        data_tracking : `pandas.core.frame.DataFrame`
            Data of the tracking event.
        """

        # Get the data from EFD
        data_controller_state, _ = await self.query_data(
            "MTRotator.logevent_controllerState",
            ["controllerState", "enabledSubstate"],
            time_start,
            time_end,
            False,
        )

        # The inPosition field is a combination of Flags_moveSuccess and
        # Flags_trackingSuccess in Simulink telemetry
        data_in_position, _ = await self.query_data(
            "MTRotator.logevent_inPosition",
            ["inPosition"],
            time_start,
            time_end,
            False,
        )

        # The tracking field is Flags_trackingSuccess in Simulink telemetry
        # The lost field is Flags_trackingLost in Simulink telemetry
        # The noNewCommand field is Flags_noNewTrackCmdError in Simulink
        # telemetry
        data_tracking, _ = await self.query_data(
            "MTRotator.logevent_tracking",
            ["tracking", "lost", "noNewCommand"],
            time_start,
            time_end,
            False,
        )

        # Add the column of time difference in seconds
        self._add_column_time_differece(data_controller_state, time_start)
        self._add_column_time_differece(data_in_position, time_start)
        self._add_column_time_differece(data_tracking, time_start)

        return data_controller_state, data_in_position, data_tracking

    async def get_data_rotation(
        self,
        time_start: Time,
        time_end: Time,
        realign_time: bool = True,
    ) -> tuple[
        DataFrame,
        numpy.typing.NDArray[np.float64],
        numpy.typing.NDArray[np.float64],
        numpy.typing.NDArray[np.float64],
    ]:
        """
        Query and return the rotation data.

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
            Data of the path generator.
        acceleration : `numpy.ndarray`
            Acceleration in deg/sec^2.
        jerk : `numpy.ndarray`
            Jerk in deg/sec^3.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        # Get the data from EFD
        data, time_operation = await self.query_data(
            "MTRotator.rotation",
            [
                "demandPosition",
                "demandVelocity",
                "demandAcceleration",
                "actualPosition",
                "actualVelocity",
                "debugActualVelocityA",
                "debugActualVelocityB",
                "timestamp",
            ],
            time_start,
            time_end,
            realign_time,
            name_timestamp="timestamp",
        )

        # Calculate the acceleration and jerk
        # Note the Simulink model had applied the low-pass filter for the
        # calculation of velocity already
        acceleration, jerk = self._calc_acceleration_and_jerk(
            time_operation, data.actualVelocity
        )

        return data, acceleration, jerk, time_operation

    def _calc_acceleration_and_jerk(
        self, time: numpy.typing.NDArray[np.float64], velocity: Series
    ) -> tuple[numpy.typing.NDArray[np.float64], numpy.typing.NDArray[np.float64]]:
        """
        Calculate the acceleration and jerk.

        Parameters
        ----------
        time : `numpy.ndarray`
            Time in second.
        velocity : `pandas.core.series.Series`
            velocity (deg/sec).

        Returns
        ----------
        acceleration : `numpy.ndarray`
            Acceleration in deg/sec^2.
        jerk : `numpy.ndarray`
            Jerk in deg/sec^3.
        """

        acceleration = np.gradient(velocity, time)
        jerk = np.gradient(acceleration, time)

        return acceleration, jerk

    async def get_data_motors(
        self,
        time_start: Time,
        time_end: Time,
        realign_time: bool = True,
    ) -> tuple[DataFrame, numpy.typing.NDArray[np.float64]]:
        """
        Get the data of motor.

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
            Data of the motor.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        data, time_operation = await self.query_data(
            "MTRotator.motors",
            [
                "raw0",
                "raw1",
                "torque0",
                "torque1",
                "current0",
                "current1",
                "busVoltage",
                "private_sndStamp",
            ],
            time_start,
            time_end,
            realign_time,
        )

        return data, time_operation

    def plot_currents(
        self, data: DataFrame, time_operation: numpy.typing.NDArray[np.float64]
    ) -> None:
        """
        Plot the data of currents.

        Parameters
        -------
        data : `pandas.core.frame.DataFrame`
            Data of the motor.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        plt.figure()
        plt.plot(time_operation, data.current0, "b")
        plt.plot(time_operation, data.current1, "r")
        plt.legend(["Motor A", "Motor B"])
        plt.title("Currents")
        plt.xlabel("Time (s)")
        plt.ylabel("Current (A)")

        plt.show()

    def plot_motor_encoders(
        self, data: DataFrame, time_operation: numpy.typing.NDArray[np.float64]
    ) -> None:
        """
        Plot the data of motor encoders.

        Parameters
        -------
        data : `pandas.core.frame.DataFrame`
            Data of the motor.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        plt.figure()
        plt.plot(time_operation, data.raw0, "b")
        plt.plot(time_operation, data.raw1, "r--")
        plt.legend(["Motor A", "Motor B"])
        plt.title("Motor Encoders")
        plt.xlabel("Time (s)")
        plt.ylabel("Raw Value")

        plt.show()

    def plot_velocity_linear_encoders(
        self, data: DataFrame, time_operation: numpy.typing.NDArray[np.float64]
    ) -> None:
        """
        Plot the velocity data of linear encoders.

        Parameters
        -------
        data : `pandas.core.frame.DataFrame`
            Data of the rotation.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        plt.figure()
        plt.plot(time_operation, data.demandVelocity, "b")
        plt.plot(time_operation, data.debugActualVelocityA, "g--")
        plt.plot(time_operation, data.debugActualVelocityB, "r--")
        plt.legend(["Demand Velocity", "Linear A", "Linear B"])
        plt.title("Linear Encoder Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (deg/s)")

        plt.show()

    def plot_path_generator(
        self,
        data: DataFrame,
        acceleration: numpy.typing.NDArray[np.float64],
        jerk: numpy.typing.NDArray[np.float64],
        time_operation: numpy.typing.NDArray[np.float64],
    ) -> None:
        """
        Plot the data of path generator.

        Parameters
        -------
        data : `pandas.core.frame.DataFrame`
            Data of the path generator.
        acceleration : `numpy.ndarray`
            Acceleration in deg/sec^2.
        jerk : `numpy.ndarray`
            Jerk in deg/sec^3.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        plt.figure()
        plt.subplot(5, 1, 1)
        plt.plot(time_operation, data.demandPosition)
        plt.plot(time_operation, data.actualPosition, "x-")
        plt.title("Path Generator")
        plt.ylabel("P")

        plt.subplot(5, 1, 2)
        plt.plot(time_operation, (data.demandPosition - data.actualPosition) * 3600)
        plt.ylabel("dP (arcsec)")

        plt.subplot(5, 1, 3)
        plt.plot(time_operation, data.demandVelocity)
        plt.plot(time_operation, data.actualVelocity, "x-")
        plt.ylabel("V")

        plt.subplot(5, 1, 4)
        plt.plot(time_operation, data.demandAcceleration)
        plt.plot(time_operation, acceleration, "x-")
        plt.ylabel("A")

        plt.subplot(5, 1, 5)
        plt.plot(time_operation, jerk)
        plt.ylabel("J")
        plt.xlabel("Time (s)")

        plt.show()

        print("Units: P (deg), dP (arcsec), V (deg/s), A (deg/s^2), J (deg/s^3)")

    def plot_event_in_position(
        self,
        data_rotation: DataFrame,
        data_in_position: DataFrame,
        time_operation: numpy.typing.NDArray[np.float64],
        threshold: float | None = None,
    ) -> None:
        """
        Plot the inPosition events.

        The vertical green line means the inPosition=True, otherwise the
        vertical red line.

        Parameters
        ----------
        data_rotation : `pandas.core.frame.DataFrame`
            Data of the rotation.
        data_in_position : `pandas.core.frame.DataFrame`
            Data of the inPosition event.
        time_operation : `numpy.ndarray`
            Operation time.
        threshold : `float` or None, optional
            Threshold of the inPosition event in degree. (the default is None)
        """

        plt.figure()

        plt.subplot(2, 1, 1)
        plt.plot(time_operation, data_rotation.demandPosition)
        plt.plot(time_operation, data_rotation.actualPosition, "x-")

        # Plot the inPosition event
        if hasattr(data_in_position, "inPosition"):
            for inPosition, time_inPosition in zip(
                data_in_position.inPosition, data_in_position.time_difference
            ):
                if inPosition:
                    plt.axvline(x=time_inPosition, color="g")
                else:
                    plt.axvline(x=time_inPosition, color="r")

        plt.legend(["Demand Position", "Actual Position"])
        plt.title("inPosition=True : green, inPosition=False : red")
        plt.ylabel("P (deg)")

        plt.subplot(2, 1, 2)

        plt.plot(
            time_operation,
            data_rotation.demandPosition - data_rotation.actualPosition,
        )

        if threshold is not None:
            plt.axhline(y=threshold, color="g")
            plt.axhline(y=-threshold, color="g")

            plt.legend(["Position Error", "+ Threshold", "- Threshold"])

        plt.xlabel("Time (s)")
        plt.ylabel("Position Error (deg)")

        plt.show()

    def plot_event_tracking(
        self,
        data_rotation: DataFrame,
        data_tracking: DataFrame,
        time_operation: numpy.typing.NDArray[np.float64],
    ) -> None:
        """
        Plot the tracking events.

        The vertical yellow line means the tracking=True.
        The vertical cyan line means the tracking=False.
        The vertical black line means the noNewCommand=True.

        Parameters
        ----------
        data_rotation : `pandas.core.frame.DataFrame`
            Data of the rotation.
        time_operation : `numpy.ndarray`
            Operation time.
        data_tracking : `pandas.core.frame.DataFrame`
            Data of the tracking event.
        """

        plt.figure()
        plt.plot(time_operation, data_rotation.demandPosition)
        plt.plot(time_operation, data_rotation.actualPosition, "x-")

        # Plot the tracking event
        for tracking, noNewCommand, time_happen in zip(
            data_tracking.tracking,
            data_tracking.noNewCommand,
            data_tracking.time_difference,
        ):
            if tracking:
                plt.axvline(x=time_happen, color="y")
            else:
                plt.axvline(x=time_happen, color="c")

            if noNewCommand:
                plt.axvline(x=time_happen, color="k")

        plt.legend(["Demand Position", "Actual Position"])

        plt.title(
            "tracking=True : yellow, tracking=False : cyan, noNewCommand=True : black"
        )

        plt.xlabel("Time (s)")
        plt.ylabel("P (deg)")

        plt.show()

    def plot_command_track(
        self,
        data_rotation: DataFrame,
        data_track: DataFrame,
        time_operation_track: numpy.typing.NDArray[np.float64],
        time_check_start: float,
        time_check_end: float,
    ) -> None:
        """Plot the track command.

        Parameters
        ----------
        data_rotation : `pandas.core.frame.DataFrame`
            Data of the rotation.
        data_track : `pandas.core.frame.DataFrame`
            Data of the track command.
        time_operation_track : `numpy.ndarray`
            Operation time of the track command.
        time_check_start : `float`
            Time to check the start.
        time_check_end : `float`
            Time to check the end.
        """

        indices_check = np.where(
            np.logical_and(
                time_operation_track >= time_check_start,
                time_operation_track <= time_check_end,
            )
        )[0]

        plt.figure()
        plt.plot(
            time_operation_track[indices_check],
            data_rotation.demandPosition.iloc[indices_check],
        )
        plt.xlabel("Time (sec)")
        plt.ylabel("P (deg)")

        # Plot the tracking command
        for angle, time_track in zip(data_track.angle, data_track.time_difference):
            if (time_check_start <= time_track <= time_check_end) and (angle != 0):
                plt.axvline(x=time_track, color="r")

        plt.show()

    def analyze_tracking_targets(
        self,
        data: DataFrame,
        time_operation: numpy.typing.NDArray[np.float64],
        list_time_start: list[float],
        list_time_end: list[float],
    ) -> None:
        """
        Analyze the tracked targets and calculate the RMS of the position
        error.

        Parameters
        -------
        data : `pandas.core.frame.DataFrame`
            Data of the path generator.
        time_operation : `numpy.ndarray`
            Operation time.
        list_time_start : `list` [`float`]
            List of the start time in second.
        list_time_end : `list` [`float`]
            List of the end time in second.
        """

        rms_positions = list()
        for time_start, time_end in zip(list_time_start, list_time_end):
            rms_position = self.analyze_tracking_target_single(
                data, time_operation, time_start, time_end, show_figure=False
            )
            rms_positions.append(rms_position)
            print("\n")

        rms_positions_overall = np.sqrt(np.mean(np.array(rms_positions) ** 2))

        digit_after_decimal = 3
        print(
            "The overall RMS of position error is "
            f"{round(rms_positions_overall, digit_after_decimal)} arcsec."
        )

    def analyze_tracking_target_single(
        self,
        data: DataFrame,
        time_operation: numpy.typing.NDArray[np.float64],
        time_start: float,
        time_end: float,
        show_figure: bool = True,
    ) -> float:
        """
        Analyze the single tracked target and calculate the RMS of the position
        error.

        Parameters
        -------
        data : `pandas.core.frame.DataFrame`
            Data of the path generator.
        time_operation : `numpy.ndarray`
            Operation time.
        time_start : `float`
            Start time in second.
        time_end : `float`
            End time in second.
        show_figure : `bool`, optional
            Show the figure or not. (the default is True)

        Returns
        -------
        rms_position : `float`
            RMS of the position error.
        """

        # Get the position data within the time range (in arcsec)
        indices = np.where(
            np.logical_and(time_operation >= time_start, time_operation <= time_end)
        )[0]
        position_demand = np.array(data.demandPosition.iloc[indices]) * 3600
        position_actual = np.array(data.actualPosition.iloc[indices]) * 3600

        # Analyze the data
        diff_position = position_demand - position_actual
        rms_position = np.sqrt(np.mean(diff_position**2))

        # Print the information
        digit_after_decimal = 3
        print(f"Analyze the data between {time_start} sec and {time_end} sec.")
        print(f"There are {len(indices)} points.")
        print(
            "The max of position error is "
            f"{round(np.max((np.abs(diff_position))), digit_after_decimal)} arcsec."
        )
        print(
            f"The RMS of position error is {round(rms_position, digit_after_decimal)} arcsec."
        )

        # Plot the figure
        if show_figure:
            plt.figure()
            plt.title("Position Error")
            plt.plot(time_operation[indices], diff_position)
            plt.xlabel("Time (s)")
            plt.ylabel("dP (arcsec)")

        return rms_position

    def calculate_frequency(
        self,
        data: DataFrame,
        time_operation: numpy.typing.NDArray[np.float64],
        time_start: float,
        time_end: float,
        start_position: float,
        tracking_velocity: float,
    ):
        """
        Calculate the oscillation frequency of tracking trajectory: position
        and velocity.

        Parameters
        -------
        data : `pandas.core.frame.DataFrame`
            Data of the path generator.
        time_operation : `numpy.ndarray`
            Operation time.
        time_start : `float`
            Start time in second.
        time_end : `float`
            End time in second.
        start_position : `float`
            Start position of target.
        tracking_velocity : `float`
            Tracking velocity in degree/second.

        Returns
        -------
        frequency_position : `float`
            Oscillation frequency of position in Hz.
        frequency_velocity : `float`
            Oscillation frequency of velocity in Hz.
        """

        # Get the position data within the time range (in arcsec)
        indices = np.where(
            np.logical_and(time_operation >= time_start, time_operation <= time_end)
        )[0]
        demand_position = (
            np.array(data.demandPosition.iloc[indices])
            - start_position
            - tracking_velocity * time_operation[indices]
        )

        dt = time_operation[1] - time_operation[0]
        data_fft_position, frequency_fft_position = self._calculate_fft(
            demand_position, dt
        )

        # Get the velocity
        demand_velocity = np.array(data.demandVelocity.iloc[indices])
        data_fft_velocity, frequency_fft_velocity = self._calculate_fft(
            demand_velocity, dt
        )

        num_position = len(demand_position)
        num_velocity = len(demand_velocity)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title("Demand position")
        plt.plot(frequency_fft_position, data_fft_position[0 : num_position // 2], "o-")

        plt.subplot(2, 1, 2)
        plt.title("Demand velocity")
        plt.plot(frequency_fft_velocity, data_fft_velocity[0 : num_velocity // 2], "o-")
        plt.xlabel("Frequency (Hz)")

    def _calculate_fft(
        self, data: numpy.typing.NDArray[np.float64], dt: float
    ) -> tuple[numpy.typing.NDArray[np.float64], numpy.typing.NDArray[np.float64]]:
        """Calculate FFT.

        Parameters
        ----------
        data : `numpy.ndarray`
            Data.
        dt : `float`
            Delta T in seconds.

        Returns
        -------
        data_fft : `numpy.ndarray`
            FFT data.
        frequency_fft : `numpy.ndarray`
            Frequency in FFT data (in Hz).
        """

        data_fft = fft(data)

        num = len(data)
        frequency_fft = fftfreq(num, dt)[: num // 2]

        return np.abs(data_fft), frequency_fft
