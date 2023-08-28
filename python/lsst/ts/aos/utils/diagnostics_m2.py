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

__all__ = ["DiagnosticsM2"]

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
from astropy.time.core import Time
from lsst.ts.m2com import NUM_ACTUATOR, NUM_TANGENT_LINK, MockControlClosedLoop
from lsst_efd_client import EfdClient
from matplotlib.axes import SubplotBase
from pandas.core.frame import DataFrame


class DiagnosticsM2:
    """M2 diagnostics class to query and plot the data.

    Parameters
    ----------
    is_summit : `bool`, optional
        This is running on the summit or not. (the default is True)

    Attributes
    ----------
    efd_client : `lsst_efd_client.efd_helper.EfdClient`
        Engineer facility database (EFD) client.
    """

    def __init__(self, is_summit: bool = True) -> None:
        self.efd_client = self._retrieve_efd_client(is_summit)

    def _retrieve_efd_client(self, is_summit: bool) -> EfdClient:
        """
        Retrieve a client to engineer facility database (EFD).

        Parameters
        ----------
        is_summit : `bool`
            This is running on the summit or not. If not, the returned object
            will point to the test stand at Tucson.

        Returns
        -------
        `EfdClient`
            The interface object between the Nublado and summit/Tucson EFD.
        """

        efd_name = "summit_efd" if is_summit else "tucson_teststand_efd"
        return EfdClient(efd_name)

    async def get_data_position(
        self,
        position_telemetry_name: str,
        time_start: Time,
        time_end: Time,
        realign_time: bool = True,
    ) -> tuple[DataFrame, numpy.typing.NDArray[np.float64]]:
        """
        Query and return the position data based on the hardpoints or
        independent measurement system (IMS).

        Parameters
        ----------
        position_telemetry_name : `str`
            Position telemetry name.
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
            Position data.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        data, time_operation = await self._query_data(
            position_telemetry_name,
            ["x", "y", "z", "xRot", "yRot", "zRot", "private_sndStamp"],
            time_start,
            time_end,
            realign_time,
        )
        return data, time_operation

    async def _query_data(
        self,
        name: str,
        fields: list[str],
        time_start: Time,
        time_end: Time,
        realign_time: bool,
    ) -> tuple[DataFrame, numpy.typing.NDArray[np.float64]]:
        """Query the EFD data.

        Parameters
        ----------
        name : `str`
            Topic name.
        fields : `list`
            Fields.
        time_start : `astropy.time.core.Time`
            Start time.
        time_end : `astropy.time.core.Time`
            End time.
        realign_time : `bool`
            Realign the timestamp to origin (0-based) or not.

        Returns
        -------
        data : `pandas.core.frame.DataFrame`
            Position data.
        time_operation : `numpy.ndarray`
            Operation time.
        """
        data = await self.efd_client.select_time_series(
            f"lsst.sal.MTM2.{name}",
            fields=fields,
            start=time_start,
            end=time_end,
        )

        # Realign the time origin to 0
        name_timestamp = "private_sndStamp"
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

    async def get_data_net_force(
        self,
        time_start: Time,
        time_end: Time,
        realign_time: bool = True,
    ) -> tuple[DataFrame, numpy.typing.NDArray[np.float64]]:
        """
        Query and return the net force data.

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
            Net force data.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        data, time_operation = await self._query_data(
            "netForcesTotal",
            ["fx", "fy", "fz", "private_sndStamp"],
            time_start,
            time_end,
            realign_time,
        )
        return data, time_operation

    async def get_data_force_balance(
        self,
        time_start: Time,
        time_end: Time,
        realign_time: bool = True,
    ) -> tuple[DataFrame, numpy.typing.NDArray[np.float64]]:
        """
        Query and return the force balance data (based on the hardpoint
        correction).

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
            Force balance data.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        data, time_operation = await self._query_data(
            "forceBalance",
            ["fx", "fy", "fz", "mx", "my", "mz", "private_sndStamp"],
            time_start,
            time_end,
            realign_time,
        )
        return data, time_operation

    async def get_data_net_moment(
        self,
        time_start: Time,
        time_end: Time,
        realign_time: bool = True,
    ) -> tuple[DataFrame, numpy.typing.NDArray[np.float64]]:
        """
        Query and return the net moment data.

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
            Net moment data.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        data, time_operation = await self._query_data(
            "netMomentsTotal",
            ["mx", "my", "mz", "private_sndStamp"],
            time_start,
            time_end,
            realign_time,
        )
        return data, time_operation

    async def get_data_net_displacement(
        self, time_start: Time, time_end: Time
    ) -> numpy.typing.NDArray[np.float64]:
        """
        Query and return the actuator's net displacement data.

        Parameters
        ----------
        time_start : `astropy.time.core.Time`
            Start time.
        time_end : `astropy.time.core.Time`
            End time.

        Returns
        -------
        displacements : `numpy.ndarray`
            Actuator's net displacement.
        """

        # Get the position of axial actuators
        num_axial = NUM_ACTUATOR - NUM_TANGENT_LINK
        fields_axial = [f"position{idx}" for idx in range(num_axial)]
        data_axial, _ = await self._query_data(
            "axialEncoderPositions",
            fields_axial,
            time_start,
            time_end,
            False,
        )

        # Get the position of tangent links
        fields_tangent = [f"position{idx}" for idx in range(NUM_TANGENT_LINK)]
        data_tangent, _ = await self._query_data(
            "tangentEncoderPositions",
            fields_tangent,
            time_start,
            time_end,
            False,
        )

        displacements = np.zeros(NUM_ACTUATOR)
        for idx in range(NUM_ACTUATOR):
            position = (
                getattr(data_axial, f"position{idx}")
                if idx < num_axial
                else getattr(data_tangent, f"position{idx-num_axial}")
            )
            displacements[idx] = position[-1] - position[0]

        return displacements

    async def get_data_force(
        self, time_start: Time, time_end: Time
    ) -> tuple[dict, dict]:
        """
        Query and return the actuator's force data.

        Parameters
        ----------
        time_start : `astropy.time.core.Time`
            Start time.
        time_end : `astropy.time.core.Time`
            End time.

        Returns
        -------
        `dict`
            Force data of the axial actuators.
        `dict`
            Force data of the tangent links.
        """

        # Prepare the fields
        components = [
            "lutGravity",
            "lutTemperature",
            "applied",
            "measured",
            "hardpointCorrection",
        ]

        num_components = len(components)
        num_axial = NUM_ACTUATOR - NUM_TANGENT_LINK

        fields_axial = self._get_fields_array(components, [num_axial] * num_components)
        fields_tangent = self._get_fields_array(
            components, [NUM_TANGENT_LINK] * num_components
        )

        # Query the data
        data_axial, _ = await self._query_data(
            "axialForce",
            fields_axial + ["private_sndStamp"],
            time_start,
            time_end,
            False,
        )
        data_tangent, _ = await self._query_data(
            "tangentForce",
            fields_tangent + ["private_sndStamp"],
            time_start,
            time_end,
            False,
        )

        data_collected_axial = self._collect_data_array(
            data_axial, components, [num_axial] * num_components
        )
        data_collected_tangent = self._collect_data_array(
            data_tangent, components, [NUM_TANGENT_LINK] * num_components
        )

        return self._set_force_error(data_collected_axial), self._set_force_error(
            data_collected_tangent
        )

    def _get_fields_array(self, components: list[str], numbers: list[int]) -> list[str]:
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

    def _collect_data_array(
        self, data_raw: DataFrame, components: list[str], numbers: list[int]
    ) -> dict:
        """Collect the array data.

        Parameters
        ----------
        data_raw : `DataFrame`
            Raw data.
        components : `list`
            List of the components.
        numbers : `list`
            List of the number of array elements.

        Returns
        -------
        data_collected : `dict`
            Collected array data.
        """
        num_row = data_raw.shape[0]

        data_collected = dict()
        for component, number in zip(components, numbers):
            values = np.zeros((num_row, number))
            for idx in range(number):
                values[:, idx] = np.array(getattr(data_raw, f"{component}{idx}"))

            data_collected[component] = values

        name_internal_timestamp = "private_sndStamp"
        if ("timestamp" not in components) and hasattr(
            data_raw, name_internal_timestamp
        ):
            data_collected["timestamp"] = getattr(data_raw, name_internal_timestamp)

        return data_collected

    def _set_force_error(self, data_force: dict) -> dict:
        """Set the force error.

        Parameters
        ----------
        data_force : `dict`
            Force data.

        Returns
        -------
        `dict`
            Updated force data with the force error.
        """

        data_force["error"] = (
            data_force["lutGravity"]
            + data_force["lutTemperature"]
            + data_force["hardpointCorrection"]
            + data_force["applied"]
            - data_force["measured"]
        )
        return data_force

    async def get_data_temperature(self, time_start: Time, time_end: Time) -> dict:
        """
        Query and return the temperature data.

        Parameters
        ----------
        time_start : `astropy.time.core.Time`
            Start time.
        time_end : `astropy.time.core.Time`
            End time.

        Returns
        -------
        `dict`
            Temperature data.
        """

        # Prepare the fields
        components = ["ring", "intake", "exhaust"]
        numbers = [12, 2, 2]
        fields_temperature = self._get_fields_array(components, numbers)

        # Query the data
        data, _ = await self._query_data(
            "temperature",
            fields_temperature + ["private_sndStamp"],
            time_start,
            time_end,
            False,
        )

        return self._collect_data_array(data, components, numbers)

    async def get_data_zenith_angle(
        self,
        time_start: Time,
        time_end: Time,
        realign_time: bool = True,
    ) -> tuple[DataFrame, numpy.typing.NDArray[np.float64]]:
        """
        Query and return the zenith angle data.

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
            Zenith angle data.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        data, time_operation = await self._query_data(
            "zenithAngle",
            ["measured", "inclinometerProcessed", "private_sndStamp"],
            time_start,
            time_end,
            realign_time,
        )
        return data, time_operation

    def get_timestamp_closest_processed_inclinometer(
        self,
        data_zenith_angle: DataFrame,
        processed_inclinometer: float,
        timestamps_zenith_angle: numpy.typing.NDArray[np.float64],
    ) -> float:
        """Get the timestamp of the closest processed inclinometer.

        Parameters
        ----------
        data_zenith_angle : `DataFrame`
            Zenith angle data.
        processed_inclinometer : `float`
            Processed inclinometer in degree.
        timestamps_zenith_angle : `numpy.ndaArray`
            Timestamps of the zenith angle.

        Returns
        -------
        `float`
            Timestamp of the closest processed inclinometer
        """

        processed_inclinometers = np.array(data_zenith_angle.inclinometerProcessed)
        idx = np.abs(processed_inclinometers - processed_inclinometer).argmin()

        return timestamps_zenith_angle[idx]

    def get_index_closest_timestamp(
        self, timestamps: numpy.typing.NDArray[np.float64], timestamp: float
    ) -> int:
        """Get the index of closest timestamp.

        Parameters
        ----------
        timestamps : `numpy.ndarray`
            Timestamps (in data A).
        timestamp : `float`
            Specific timestamp (exists in data A or B).

        Returns
        -------
        `int`
            Index of closed timestamp.
        """

        return np.abs(timestamps - timestamp).argmin()

    def get_xy_actuators(
        self,
        control_closed_loop: MockControlClosedLoop,
    ) -> numpy.typing.NDArray[np.float64]:
        """Get the (x, y) position of actuators.

        Parameters
        ----------
        control_closed_loop : `lsst.ts.m2.com.MockControlClosedLoop`
            Mock control closed loop instance.

        Returns
        -------
        xy_actuators : `numpy.ndarray`
            Actuator x, y positions in meter.
        """
        # Axial actuators
        xy_axial = np.array(control_closed_loop.get_actuator_location_axial())

        # Tangent links
        radius = control_closed_loop.get_radius()
        angle_radian = np.deg2rad(control_closed_loop.get_actuator_location_tangent())

        xy_tangent = np.zeros([NUM_TANGENT_LINK, 2])
        xy_tangent[:, 0] = radius * np.cos(angle_radian)
        xy_tangent[:, 1] = radius * np.sin(angle_radian)

        xy_actuators = np.zeros([NUM_ACTUATOR, 2])
        xy_actuators[:-NUM_TANGENT_LINK, :] = xy_axial
        xy_actuators[-NUM_TANGENT_LINK:, :] = xy_tangent

        return xy_actuators

    def set_lut_force_theoretical(
        self,
        control_closed_loop: MockControlClosedLoop,
        data_axial: dict,
        data_tangent: dict,
        data_temperature: dict,
        data_zenith_angle: DataFrame,
        time_operation_angle: numpy.typing.NDArray[np.float64],
    ) -> tuple[dict, dict]:
        """Set the theoretical look-up table (LUT) forces.

        Parameters
        ----------
        control_closed_loop : `lsst.ts.m2.com.MockControlClosedLoop`
            Mock control closed loop instance.
        data_axial : `dict`
            Data of the axial force.
        data_tangent : `dict`
            Data of the tangent force.
        data_temperature : `dict`
            Data of the temperature.
        data_zenith_angle : `pandas.core.frame.DataFrame`
            Zenith angle data.
        time_operation_angle : `numpy.ndarray`
            Operation time of the angle.

        Returns
        -------
        data_axial : `dict`
            Updated data of the axial force.
        data_tangent : `dict`
            Updated data of the tangent force.
        """

        data_axial, data_tangent = self._set_lut_force_theoretical_individual(
            control_closed_loop,
            data_axial,
            data_tangent,
            data_temperature,
            data_zenith_angle,
            time_operation_angle,
            update_axial=True,
        )
        data_axial, data_tangent = self._set_lut_force_theoretical_individual(
            control_closed_loop,
            data_axial,
            data_tangent,
            data_temperature,
            data_zenith_angle,
            time_operation_angle,
            update_axial=False,
        )
        return data_axial, data_tangent

    def _set_lut_force_theoretical_individual(
        self,
        control_closed_loop: MockControlClosedLoop,
        data_axial: dict,
        data_tangent: dict,
        data_temperature: dict,
        data_zenith_angle: DataFrame,
        time_operation_angle: numpy.typing.NDArray[np.float64],
        update_axial: bool = True,
    ) -> tuple[dict, dict]:
        """Set the individual theoretical LUT forces.

        Parameters
        ----------
        control_closed_loop : `lsst.ts.m2.com.MockControlClosedLoop`
            Mock control closed loop instance.
        data_axial : `dict`
            Data of the axial force.
        data_tangent : `dict`
            Data of the tangent force.
        data_temperature : `dict`
            Data of the temperature.
        data_zenith_angle : `pandas.core.frame.DataFrame`
            Zenith angle data.
        time_operation_angle : `numpy.ndarray`
            Operation time of the angle.
        update_axial : `bool`, optional
            True if update the data of axial force. Otherwise, update the data
            of tangent force. (the default is True)

        Returns
        -------
        data_axial : `dict`
            Updated data of the axial force.
        data_tangent : `dict`
            Updated data of the tangent force.
        """

        # The main data will be updated.
        # The minor data is used to support the calculation.
        data_main = data_axial if update_axial else data_tangent
        data_minor = data_tangent if update_axial else data_axial

        shape = data_main["lutGravity"].shape
        lut_gravity_theoretical = np.zeros(shape)

        if update_axial:
            lut_temperature_theoretical = np.zeros(shape)

        for idx, force_measured in enumerate(data_main["measured"]):
            # Set the temperatures
            temperature_ring = self._find_nearest_value_based_on_timestamp(
                data_temperature["ring"],
                np.array(data_temperature["timestamp"]),
                data_main["timestamp"][idx],
            )
            temperature_intake = self._find_nearest_value_based_on_timestamp(
                data_temperature["intake"],
                np.array(data_temperature["timestamp"]),
                data_main["timestamp"][idx],
            )
            temperature_exhaust = self._find_nearest_value_based_on_timestamp(
                data_temperature["exhaust"],
                np.array(data_temperature["timestamp"]),
                data_main["timestamp"][idx],
            )

            control_closed_loop.temperature["ring"] = temperature_ring.tolist()  # type: ignore
            control_closed_loop.temperature["intake"] = temperature_intake.tolist()  # type: ignore
            control_closed_loop.temperature["exhaust"] = temperature_exhaust.tolist()  # type: ignore

            # Set the measured force
            force_measured_minor = self._find_nearest_value_based_on_timestamp(
                data_minor["measured"],
                np.array(data_minor["timestamp"]),
                data_main["timestamp"][idx],
            )

            if update_axial:
                control_closed_loop.set_measured_forces(
                    force_measured, force_measured_minor
                )
            else:
                control_closed_loop.set_measured_forces(
                    force_measured_minor, force_measured
                )

            # Calculate the LUT force. Note the "processed" inclinometer angle
            # is used in the LUT calculation.
            lut_angle = self._find_nearest_value_based_on_timestamp(
                data_zenith_angle["inclinometerProcessed"],
                time_operation_angle,
                data_main["timestamp"][idx],
            )

            control_closed_loop.calc_look_up_forces(lut_angle)

            if update_axial:
                forces_theoretical = control_closed_loop.axial_forces
            else:
                forces_theoretical = control_closed_loop.tangent_forces

            # Update the data
            lut_gravity_theoretical[idx, :] = forces_theoretical["lutGravity"]

            # Tangent link has no temperature correction
            if update_axial:
                lut_temperature_theoretical[idx, :] = forces_theoretical[
                    "lutTemperature"
                ]

        data_main["lutGravityTheoretical"] = lut_gravity_theoretical

        if update_axial:
            data_main["lutTemperatureTheoretical"] = lut_temperature_theoretical

        return data_axial, data_tangent

    def _find_nearest_value_based_on_timestamp(
        self,
        values: numpy.typing.NDArray[np.float64],
        timestamps: numpy.typing.NDArray[np.float64],
        timestamp: float,
    ) -> float | numpy.typing.NDArray[np.float64]:
        """Find the nearest value based on the timestamp.

        Parameters
        ----------
        values : `numpy.ndarray`
            Values.
        timestamps : `numpy.ndarray`
            Timestamps.
        timestamp : `float`
            Timestamp.

        Returns
        -------
        `float` or `numpy.ndarray`
            Nearest value.
        """
        idx = (np.abs(timestamps - timestamp)).argmin()
        return values[idx]

    async def get_data_power_status(
        self,
        time_start: Time,
        time_end: Time,
        realign_time: bool = True,
    ) -> tuple[DataFrame, numpy.typing.NDArray[np.float64]]:
        """
        Query and return the power status data.

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
            Position data.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        data, time_operation = await self._query_data(
            "powerStatus",
            [
                "motorVoltage",
                "motorCurrent",
                "commVoltage",
                "commCurrent",
                "private_sndStamp",
            ],
            time_start,
            time_end,
            realign_time,
        )
        return data, time_operation

    async def get_data_step_axial(
        self, time_start: Time, time_end: Time, realign_time: bool = True
    ) -> tuple[numpy.typing.NDArray[int], numpy.typing.NDArray[np.float64]]:
        """
        Query and return the axial actuator's step data.

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
        data : `numpy.ndarray`
            Step data.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        # Prepare the fields
        component = ["steps"]
        num_axial = NUM_ACTUATOR - NUM_TANGENT_LINK
        fields_axial = self._get_fields_array(component, [num_axial])

        # Query the data
        data, time_operation = await self._query_data(
            "axialActuatorSteps",
            fields_axial + ["private_sndStamp"],
            time_start,
            time_end,
            realign_time,
        )

        # Only 1 component
        data_collected = self._collect_data_array(data, component, [num_axial])

        return data_collected["steps"], time_operation

    async def get_data_step_tangent(
        self, time_start: Time, time_end: Time, realign_time: bool = True
    ) -> tuple[numpy.typing.NDArray[int], numpy.typing.NDArray[np.float64]]:
        """
        Query and return the tangential actuator's step data.

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
        data : `numpy.ndarray`
            Step data.
        time_operation : `numpy.ndarray`
            Operation time.
        """

        # Prepare the fields
        component = ["steps"]
        fields_tangent = self._get_fields_array(component, [NUM_TANGENT_LINK])

        # Query the data
        data, time_operation = await self._query_data(
            "tangentActuatorSteps",
            fields_tangent + ["private_sndStamp"],
            time_start,
            time_end,
            realign_time,
        )

        # Only 1 component
        data_collected = self._collect_data_array(data, component, [NUM_TANGENT_LINK])

        return data_collected["steps"], time_operation

    def draw_values(
        self,
        xy_actuators: numpy.typing.NDArray[np.float64],
        values_axial: numpy.typing.NDArray[np.float64 | int],
        values_tangent: numpy.typing.NDArray[np.float64 | int],
        max_marker_size: int = 300,
        title: str = "",
        hardpoints: list[int] | None = None,
    ) -> None:
        """Draw the values on x, y map.

        Parameters
        ----------
        xy_actuators : `numpy.ndarray`
            X, Y positions of actuators.
        values_axial : `numpy.ndarray`
            Axial actuator values.
        values_tangent : `numpy.ndarray`
            Tangent actuator values.
        max_marker_size : `float`, optional
            Maximum marker size. (the defautl is 300)
        title : `str`, optional
            Title. (the default is "")
        hardpoints : `list` or None, optional
            Ordered 0-based six hardpoints. The first three are the axial
            actuators and the latters are the tangent links. If not None, they
            will be labeled on the drawing. (the default is None)
        """

        max_value_axial = np.max(np.abs(values_axial))
        max_value_tangent = np.max(np.abs(values_tangent))

        magnification_axial = (
            max_marker_size / max_value_axial if max_value_axial != 0 else 1
        )
        magnification_tangent = (
            max_marker_size / max_value_tangent if max_value_tangent != 0 else 1
        )

        fig, ax = plt.subplots(2, 1, figsize=(5, 10))

        # Draw the axial actuators
        ax[0].set_title("Axial Actuators")
        img = ax[0].scatter(
            xy_actuators[:-NUM_TANGENT_LINK, 0],
            xy_actuators[:-NUM_TANGENT_LINK, 1],
            s=np.abs(values_axial) * magnification_axial,
            c=values_axial,
            vmin=min(values_axial),
            vmax=max(values_axial),
        )
        fig.colorbar(img, ax=ax[0])

        plt.ylabel("Y position (m)")

        # Draw the tangent links
        ax[1].set_title("Tangent Links")
        img = ax[1].scatter(
            xy_actuators[-NUM_TANGENT_LINK:, 0],
            xy_actuators[-NUM_TANGENT_LINK:, 1],
            s=np.abs(values_tangent) * magnification_tangent,
            c=values_tangent,
            vmin=min(values_tangent),
            vmax=max(values_tangent),
        )
        fig.colorbar(img, ax=ax[1])

        plt.xlabel("X position (m)")

        # Label the hardpoints
        if hardpoints is not None:
            for hardpoint in hardpoints:
                alias = self.get_alias_based_on_actuator_id(hardpoint)

                ax_actuator = ax[1] if alias.startswith("A") else ax[0]
                ax_actuator.annotate(alias, xy_actuators[hardpoint, :])

        ax[0].axis("equal")
        ax[1].axis("equal")

        plt.suptitle(title)

        plt.show()

    def get_alias_based_on_actuator_id(self, actuator_id: int) -> str:
        """Get the actuator's alias based on the actuator ID.

        Notes
        -----
        Actuator numbers in ring B, C, D, and A are 30, 24, 18, and 6. If the
        actuator ID is 5, the alias is B6. If it is 30, the alias is C1. The
        tangent links belong to the ring A. Note the alias is 1-based.

        Parameters
        ----------
        actuator_id : `int`
            0-based actuator ID.

        Returns
        -------
        `str`
            Actuator alias.
        """

        index = actuator_id + 1

        if actuator_id < 30:
            return f"B{index}"
        elif 30 <= actuator_id < 54:
            return f"C{index - 30}"
        elif 54 <= actuator_id < 72:
            return f"D{index - 54}"
        else:
            return f"A{index - 72}"

    def plot_positions(
        self,
        data_position: DataFrame,
        time_operation_position: numpy.typing.NDArray[np.float64],
        data_ims: DataFrame,
        time_operation_ims: numpy.typing.NDArray[np.float64],
    ) -> None:
        """
        Plot the data of positions.

        Parameters
        -------
        data_position : `pandas.core.frame.DataFrame`
            Data based on the hardpoints.
        time_operation_position : `numpy.ndarray`
            Realigned operation time in second (0-based) based on the
            hardpoints.
        data_ims : `pandas.core.frame.DataFrame`
            Data based on the independent measurement system (IMS).
        time_operation_ims : `numpy.ndarray`
            Realigned operation time in second (0-based) based on IMS.
        """

        fig, axs = plt.subplots(2, 3)

        axes = ["x", "y", "z", "xRot", "yRot", "zRot"]
        for idx, axis in enumerate(axes):
            row = 0 if idx < 3 else 1
            col = idx % 3
            self._subplot(
                axs[row, col],
                [data_position, data_ims],
                [time_operation_position, time_operation_ims],
                ["b", "r"],
                axis,
            )

        axs[0, 0].set_ylabel("Position (um)")
        axs[1, 0].set_ylabel("Angle (arcsec)")
        axs[1, 1].set_xlabel("Time (sec)")

        fig.suptitle("Position")
        fig.tight_layout()
        fig.legend(["Hardpoints", "IMS"], loc="lower right")

        plt.show()

    def _subplot(
        self,
        ax: SubplotBase,
        datas: list[DataFrame],
        time_operations: list[numpy.typing.NDArray[np.float64]],
        colors: list[str],
        component: str,
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
        """
        for data, time_operation, color in zip(datas, time_operations, colors):
            ax.plot(time_operation, getattr(data, component), color)

        ax.set_title(f"{component}")

    def plot_net_force(
        self, data: DataFrame, time_operation: numpy.typing.NDArray[np.float64]
    ) -> None:
        """
        Plot the data of net force.

        Parameters
        -------
        data : `pandas.core.frame.DataFrame`
            Data of the net force.
        time_operation : `numpy.ndarray`
            Realigned operation time in second (0-based).
        """

        fig, axs = plt.subplots(1, 3)

        components = ["fx", "fy", "fz"]
        for idx, component in enumerate(components):
            self._subplot(axs[idx], [data], [time_operation], ["b"], component)

        axs[0].set_ylabel("Force (N)")
        axs[1].set_xlabel("Time (sec)")

        fig.suptitle("Net Force")
        fig.tight_layout()

        plt.show()

    def plot_net_moment(
        self, data: DataFrame, time_operation: numpy.typing.NDArray[np.float64]
    ) -> None:
        """
        Plot the data of net moment.

        Parameters
        -------
        data : `pandas.core.frame.DataFrame`
            Data of the net moment.
        time_operation : `numpy.ndarray`
            Realigned operation time in second (0-based).
        """

        fig, axs = plt.subplots(1, 3)

        components = ["mx", "my", "mz"]
        for idx, component in enumerate(components):
            self._subplot(axs[idx], [data], [time_operation], ["b"], component)

        axs[0].set_ylabel("Moment (N * m)")
        axs[1].set_xlabel("Time (sec)")

        fig.suptitle("Net Moment")
        fig.tight_layout()

        plt.show()

    def plot_force_balance(
        self, data: DataFrame, time_operation: numpy.typing.NDArray[np.float64]
    ) -> None:
        """
        Plot the data of force balance.

        Parameters
        -------
        data : `pandas.core.frame.DataFrame`
            Force balance data.
        time_operation : `numpy.ndarray`
            Realigned operation time in second (0-based).
        """

        fig, axs = plt.subplots(2, 3)

        components = ["fx", "fy", "fz", "mx", "my", "mz"]
        for idx, component in enumerate(components):
            row = 0 if idx < 3 else 1
            col = idx % 3
            self._subplot(axs[row, col], [data], [time_operation], ["b"], component)

        axs[0, 0].set_ylabel("Force (N)")
        axs[1, 0].set_ylabel("Moment (N * m)")
        axs[1, 1].set_xlabel("Time (sec)")

        fig.suptitle("Force Balance")
        fig.tight_layout()

        plt.show()

    def plot_net_displacement(
        self,
        displacements_theory: numpy.typing.NDArray[np.float64],
        displacements_experiment: numpy.typing.NDArray[np.float64],
    ) -> None:
        """
        Plot the net displacement.

        Parameters
        -------
        displacements_theory : `numpy.ndarray`
            Theoretical displacements in um.
        displacements_experiment : `numpy.ndarray`
            Experimental displacements in um.
        """

        actuator_idx = range(NUM_ACTUATOR)

        plt.figure()

        plt.plot(actuator_idx, displacements_theory, "b")
        plt.plot(actuator_idx, displacements_experiment, "r--")

        plt.legend(["Theory", "Experiment"])

        plt.xlabel("Actuator Index")
        plt.ylabel("Displacement (um)")

        plt.title("Net Displacement")

        plt.show()

    def plot_processed_inclinometer(
        self, data: DataFrame, time_operation: numpy.typing.NDArray[np.float64]
    ) -> None:
        """
        Plot the data of processed inclinometer angle.

        Parameters
        -------
        data : `pandas.core.frame.DataFrame`
            Data of the net force.
        time_operation : `numpy.ndarray`
            Realigned operation time in second (0-based).
        """

        plt.figure()
        plt.plot(time_operation, data.inclinometerProcessed, "b")

        plt.xlabel("Time (second)")
        plt.ylabel("Angle (degree)")

        plt.title("Processed Inclinometer")

        plt.show()

    def plot_force_error(self, data_axial: dict, data_tangent: dict) -> None:
        """Plot the force error.

        Parameters
        ----------
        data_axial : `dict`
            Axial force data.
        data_tangent : `dict`
            Tangent force data.
        """

        fig, axs = plt.subplots(1, 2)

        datas = [data_axial, data_tangent]
        numbers = [NUM_ACTUATOR - NUM_TANGENT_LINK, NUM_TANGENT_LINK]
        titles = ["axial", "tangent"]
        for data, number, ax, title in zip(datas, numbers, axs, titles):
            for idx in range(number):
                ax.plot(data["timestamp"] - data["timestamp"][0], data["error"][:, idx])

            ax.set_title(title)

            ax.set_xlabel("Time (sec)")
            ax.set_ylabel("Force Error (N)")

        fig.suptitle("Force Error (LUT + applied + hardpointCorrection - measured)")
        fig.tight_layout()

        plt.show()

    def plot_lut_force_error_axial(self, data_axial: dict) -> None:
        """Plot the look-up table (LUT) force error of axial actuators.

        Parameters
        ----------
        data_axial : `dict`
            Axial force data.
        """

        fig, axs = plt.subplots(1, 2)

        numbers_axial = NUM_ACTUATOR - NUM_TANGENT_LINK
        components = ["lutGravity", "lutTemperature"]
        for ax, component in zip(axs, components):
            for idx in range(numbers_axial):
                ax.plot(
                    data_axial["timestamp"] - data_axial["timestamp"][0],
                    data_axial[component][:, idx]
                    - data_axial[f"{component}Theoretical"][:, idx],
                )

            ax.set_title(component)

            ax.set_xlabel("Time (sec)")
            ax.set_ylabel("LUT Force Error (N)")

        fig.suptitle("Axial LUT Force Error (Measurement - Theory)")
        fig.tight_layout()

        plt.show()

    def plot_lut_force_error_tangent(self, data_tangent: dict) -> None:
        """Plot the look-up table (LUT) force error of tangent actuators.

        Parameters
        ----------
        data_tangent : `dict`
            Tangent force data.
        """

        plt.figure()

        for idx in range(NUM_TANGENT_LINK):
            plt.plot(
                data_tangent["timestamp"] - data_tangent["timestamp"][0],
                data_tangent["lutGravity"][:, idx]
                - data_tangent["lutGravityTheoretical"][:, idx],
            )

        plt.xlabel("Time (sec)")
        plt.ylabel("LUT Force Error (N)")

        plt.title("Tangent LUT Force Error (Measurement - Theory)")

        plt.show()

    def print_actuator_force_error_out_threshold(
        self, data_force: dict, threshold: float, actuator_type: str
    ) -> None:
        """Print the actuator index that has the force error higher than the
        threshold.

        Parameters
        ----------
        data_force : `dict`
            Force data.
        threshold : `float`
            Force threshold in Newton.
        actuator_type : `str`
            Actuator type.
        """

        for force_error, timestamp in zip(data_force["error"], data_force["timestamp"]):
            idx = np.where(np.abs(force_error) > threshold)[0]
            if len(idx) > 0:
                print(
                    f"Error of {actuator_type} actuator {idx} > {threshold} N at time: {timestamp} second."
                )
