""" Utility functions for the pipelines. """
import hashlib
import logging
from enum import Enum
from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd
import yaml
from strictyaml import YAML
from strictyaml import load

from transaction_risk_profiler.common.enums.dataset import DataColumnsEnum
from transaction_risk_profiler.configs import settings
from transaction_risk_profiler.configs.dataset_config import DatasetCardConfig

logger = logging.getLogger(__name__)


class DatasetSplitRatioError(ValueError):
    """Exception raised when the ratio is not between 0 and 1."""

    def __init__(self, ratio_type: str, ratio: float):
        """Initialize the exception."""
        super().__init__(f"{ratio_type} ratio must be between 0 and 1, got {ratio}")


def is_id_in_set_sha256(identifier: int, ratio: float, offset: float = 0.0) -> bool:
    """
    Determine if an identifier should belong to a particular dataset set (test or holdout).

    Parameters
    ----------
    identifier : int
        The identifier to hash.
    ratio : float
        The ratio of the set.
    offset : float
        Offset to apply for selecting a different set.

    Returns
    -------
    bool
        True if the identifier should be in the set, False otherwise.
    """
    sha256 = hashlib.sha256()
    sha256.update(str(identifier).encode("utf-8"))
    hashed_id = int(sha256.hexdigest(), 16)

    if ratio + offset > 0.75:
        logger.warning(
            f"Test ratio and holdout ratio are > 0.75: {ratio + offset}. This "
            f"leaves {1 - ratio - offset} data for training."
        )

    return 2**256 * offset <= hashed_id < 2**256 * (offset + ratio)


def is_id_in_set(identifier: int, ratio: float, offset: float = 0.0) -> bool:
    """
    Determine if an identifier should belong to a particular dataset set (test or holdout).

    Parameters
    ----------
    identifier : int
        The identifier to hash.
    ratio : float
        The ratio of the set.
    offset : float
        Offset to apply for selecting a different set.

    Returns
    -------
    bool
        True if the identifier should be in the set, False otherwise.
    """
    # return crc32(np.int64(identifier)) >= 2**32 * offset and crc32(np.int64(identifier)) < (
    #     2**32 * (offset + ratio)
    # )
    if ratio + offset > 0.75:
        logger.warning(
            f"Test ratio and holdout ratio are > 0.75: {ratio + offset}. This "
            f"leaves {1 - ratio - offset} data for training."
        )
    return 2**32 * offset <= crc32(np.int64(identifier)) < (2**32 * (offset + ratio))


def split_data_with_id_hash(
    data: pd.DataFrame, test_ratio: float, id_column: str, holdout_ratio: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, None | pd.DataFrame]:
    """
    Split data into train, test, and optionally holdout sets.

    Parameters
    ----------
    data : pd.DataFrame
        The data to split.
    test_ratio : float
        The ratio of the test set.
    id_column : str
        The name of the column containing the identifier.
    holdout_ratio : float, optional
        The ratio of the holdout set, default is None.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The training, test, and optionally holdout sets.
    """
    if id_column not in data.columns and id_column != "index":
        raise ValueError(f"Column {id_column} not found in data.")

    if id_column == "index" and "index" not in data.columns:
        data.reset_index(inplace=True)

    if test_ratio < 0 or test_ratio > 1:
        raise DatasetSplitRatioError("Test", holdout_ratio)

    if holdout_ratio is not None and (holdout_ratio < 0 or holdout_ratio > 1):
        raise DatasetSplitRatioError("Holdout", holdout_ratio)

    ids = data[id_column]

    in_test_set = ids.apply(lambda id_: is_id_in_set(id_, test_ratio))
    in_train_set = ~in_test_set

    if holdout_ratio:
        in_holdout_set = ids.apply(lambda id_: is_id_in_set(id_, holdout_ratio, offset=test_ratio))
        in_train_set = ~(in_test_set | in_holdout_set)
        holdout_data = data.loc[in_holdout_set]
    else:
        holdout_data = None

    return data.loc[in_train_set], data.loc[in_test_set], holdout_data


def fetch_dataset_card(
    dataset_card_path: Path | str | None = None, dataset_card_filename: str | None = "dataset_card"
) -> YAML:
    """Parse YAML containing the package configuration."""

    if not dataset_card_path:
        if dataset_card_filename.endswith(".yaml"):
            pass
        elif not dataset_card_filename.endswith(".yml"):
            dataset_card_filename = f"{dataset_card_filename}.yml"

        dataset_card_path = Path(
            f"{settings.PROJECT_DIRECTORY}/{settings.DATASET_DIRECTORY}/{dataset_card_filename}"
        )

    if isinstance(dataset_card_path, str):
        dataset_card_path = Path(dataset_card_path)

    if not dataset_card_path.is_file():
        raise OSError(f"Did not find config file at path: {dataset_card_path}")

    if dataset_card_path:
        with open(Path(dataset_card_path)) as conf_file:
            return load(conf_file.read())


def save_as_yaml(dataset_config: DatasetCardConfig, file_path: str):
    """
    Save a DatasetCardConfig instance as a YAML file, preserving the order of keys.

    Parameters
    ----------
    dataset_config : DatasetCardConfig
        The instance of the DatasetCardConfig class to be saved.
    file_path : str
        The path where the YAML file will be saved.

    Returns
    -------
    None
    """
    # Convert the Pydantic model to a dictionary
    config_dict = dataset_config.model_dump(exclude_none=True)

    # Convert any Enums to their string representation
    for key, value in config_dict.items():
        if isinstance(value, Enum):
            config_dict[key] = value.value

    # Add additional metadata or headers if needed
    headers = {
        "name": "transactions",
        "description": "Anonymized transactions labeled as fraudulent or genuine",
        "url": None,
        "license": "CC BY-NC-SA",
    }

    # Merge the headers and the config dictionary
    final_config = {**headers, **config_dict}

    # Save the dictionary as a YAML file, preserving the order
    with open(file_path, "w") as f:
        yaml.dump(final_config, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    # Example usage
    dataset_config_example = DatasetCardConfig(
        target=DataColumnsEnum.get_target(),
        features=DataColumnsEnum.get_feature_names(),
        id_column="object_id",
        features_to_rename={DataColumnsEnum.get_target(): "target"},
        numerical_features=[
            "body_length",
            "gts",
            "name_length",
            "num_order",
            "num_payouts",
            "org_facebook",
            "org_twitter",
            "sale_duration",
            "sale_duration2",
            "user_age",
        ],
        categorical_features=[
            "channels",
            "country",
            "currency",
            "delivery_method",
            "payout_type",
        ],
        binary_features=[
            "fb_published",
            "has_analytics",
            "has_header",
            "has_logo",
            "listed",
            "show_map",
        ],
        datetime_features=[
            "approx_payout_date",
            "event_created",
            "event_end",
            "event_published",
            "event_start",
            "user_created",
        ],
        geo_features=["venue_latitude", "venue_longitude"],
        text_features=[
            "country",
            "currency",
            "description",
            "email_domain",
            "name",
            "org_desc",
            "org_name",
            "payee_name",
            "venue_address",
            "venue_country",
            "venue_name",
            "venue_state",
        ],
        list_features=["previous_payouts", "ticket_types"],
    )

    save_as_yaml(dataset_config_example, "data/dataset_card_save.yml")
