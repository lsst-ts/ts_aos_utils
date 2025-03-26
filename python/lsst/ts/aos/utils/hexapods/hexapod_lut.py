import argparse
import asyncio

import numpy as np
import pandas as pd
import yaml
from astropy.time import Time
from lsst_efd_client import EfdClient
from numpy.polynomial import Polynomial


def extract_component_data(offsets: pd.DataFrame, component: str) -> pd.DataFrame:
    """Extracts the component data from the offsets dataframe

    Parameters
    ----------
    offsets: pandas.DataFrame
        Dataframe with the offsets data from the EFD
    component: str
        Component to extract the data for. Can be 'M2', 'Camera'

    Returns
    -------
    pandas.DataFrame
        Dataframe with the offsets data for the specified component
    """

    def extract_target_angles(offsets: pd.DataFrame) -> pd.DataFrame:
        """Modifies the input DataFrame to include extracted target angles."""
        # Initialize columns to zeros
        offsets["target_elevation"] = 0.0
        offsets["target_azimuth"] = 0.0
        offsets["target_rotation"] = 0.0

        # Iterate through the DataFrame and extract angles
        for index, row in offsets.iterrows():
            parts = row["target"].split("_")
            if parts[1] == "CENTRAL":
                angles = [float(parts[2]), float(parts[3]), float(parts[4])]
            else:
                angles = [float(parts[1]), float(parts[2]), float(parts[3])]
            offsets.at[index, "target_elevation"] = angles[0]
            offsets.at[index, "target_azimuth"] = angles[1]
            offsets.at[index, "target_rotation"] = angles[2]

        return offsets

    if component == "Camera":
        offsets = offsets.loc[offsets["target"].str.contains("FrameCAM")].copy()
    elif component == "M2":
        offsets = offsets.loc[offsets["target"].str.contains("FrameM2")].copy()
    elif component == "TMA_CENTRAL":
        offsets = offsets.loc[offsets["target"].str.contains("FrameTMA_CENTRAL")].copy()

    processed_offsets = extract_target_angles(offsets)

    # Convert the target angles to um
    processed_offsets["dX"] *= 1e6
    processed_offsets["dY"] *= 1e6
    processed_offsets["dZ"] *= 1e6
    return processed_offsets


async def update_hexapod_lut(
    start_time: Time,
    end_time: Time,
    lut_type: str,
    lut_path: str,
    output_path: str,
    components: str,
    is_compensation_mode_on: bool,
    polynomial_degree: int,
    client: EfdClient,
) -> None:
    """Update the LUT file with the LaserTracker offsets data from the EFD.
    Saves a new yaml file with LUT updated values

    Parameters
    ----------
    start_time: astropy.time.Time
        Start time of the data to be queried from the EFD
    end_time: astropy.time.Time
        End time of the data to be queried from the EFD
    lut_type: str
        Type of LUT to be updated. Can be 'elevation', 'rotation', or 'azimuth'
    lut_path: str
        Path to the LUT file to be updated.
    output_path: str
        Path to the output LUT file
    components: str
        Components to update the LUT for. Can be 'M2', 'Camera', or 'M2Camera'
    is_compensation_mode_on: bool
        Was hexapod compensation mode on when taking the data.
        If so, LUT update is an increment.
    polynomial_degree: int
        Degree of the polynomial to be fitted to the data
    client: EfdClient
        Client for EFD database

    Returns
    -------
    None
    """

    # Load LUT yaml
    with open(lut_path, "r") as yaml_file:
        lut_data = yaml.safe_load(yaml_file)

    # Query Laser Tracker offsets from EFD from start_time to end_time
    print("Retrieving applied forces data from EFD...")
    fields = ["dX", "dY", "dZ", "dRX", "dRY", "target"]
    fields_sign = np.array([-1, -1, -1, 1, 1])
    offsets = await client.select_time_series(
        "lsst.sal.LaserTracker.logevent_offsetsPublish",
        fields,
        start_time,
        end_time,
    )

    # Fit LUT for all the components
    # Function to extract components from a component argument
    def extract_keywords(s: str) -> list[str]:
        return [keyword for keyword in ["M2", "Camera"] if keyword in s]

    fitted_components = extract_keywords(components)
    for component in fitted_components:
        data_camera = extract_component_data(offsets, component)

        for idx, field in enumerate(fields[:-1]):
            data_to_fit = -fields_sign[idx] * data_camera[field]
            fitted_polynomial, [residual, _, _, _] = Polynomial.fit(
                data_camera[f"target_{lut_type}"],
                data_to_fit - np.mean(data_to_fit),
                polynomial_degree,
                full=True,
            )

            if is_compensation_mode_on:
                current_coefs = np.array(
                    lut_data[f"{component.lower()}_config"][f"{lut_type}_coeffs"][idx]
                )
                updated_coefs = current_coefs + fitted_polynomial.convert().coef
            else:
                updated_coefs = fitted_polynomial.convert().coef

            lut_data[f"{component.lower()}_config"][f"{lut_type}_coeffs"][
                idx
            ] = updated_coefs.tolist()

    # Save the updated LUT file
    print("Saving LUT file")

    with open(output_path, "w") as yaml_file:
        yaml.safe_dump(lut_data, yaml_file, default_flow_style=False, sort_keys=False)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update Hexapod LUT file with Laser Tracker offsets"
    )

    parser.add_argument(
        "start_time",
        help="Start time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'",
        type=Time,
    )

    parser.add_argument(
        "end_time",
        help="End time in a valid format: \
            'YYYY-MM-DD HH:MM:SSZ'",
        type=Time,
    )

    parser.add_argument(
        "--lut_path", default=None, help="Path to the original LUT file"
    )

    parser.add_argument(
        "--output_path", default="hexapod_lut.yaml", help="Path to the output LUT file"
    )

    parser.add_argument(
        "--lut_type",
        choices=["elevation", "rotation", "azimuth"],
        default="elevation",
        help="Type of LUT to be updated",
    )

    parser.add_argument(
        "--components",
        choices=["M2", "Camera", "M2Camera"],
        default="M2Camera",
        help="Component to update the LUT for. \
            Defaults to both M2 and Camera.",
    )

    parser.add_argument(
        "--is_compensation_mode_on",
        type=bool,
        default=False,
        help="Was hexapod compensation mode on when taking \
            the data. If so, LUT update is an increment.",
    )

    parser.add_argument(
        "--polynomial-degree",
        default=5,
        type=int,
        help="Degree of the polynomial to be fitted to the data",
    )

    parser.add_argument(
        "--efd",
        default="usdf_efd",
        help="EFD name. Defaults to usdf_efd",
    )

    return parser.parse_args()


async def process_data(args: argparse.Namespace) -> None:
    print("Connecting to EFD", args.efd)
    client = EfdClient(args.efd)

    await update_hexapod_lut(
        args.start_time,
        args.end_time,
        args.lut_type,
        args.lut_path,
        args.output_path,
        args.components,
        args.is_compensation_mode_on,
        args.polynomial_degree,
        client,
    )


def run() -> None:
    """Main function to run the process_data function."""
    asyncio.run(process_data(parse_arguments()))
