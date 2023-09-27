""" This module contains the DatasetTransporter class.

To run this module independently, use the following command:
python src/transaction_risk_profiler/io/dataset_transporter.py
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from attrs import define
from attrs import field
from strictyaml import YAML

from transaction_risk_profiler.configs import settings
from transaction_risk_profiler.configs.dataset_config import DatasetCardConfig
from transaction_risk_profiler.io.dataset_utils import fetch_dataset_card
from transaction_risk_profiler.io.dataset_utils import split_data_with_id_hash

logger = logging.getLogger(__name__)


@define
class DataTransporter:
    """
    This class facilitates the transportation of data into various Python
    machine learning algorithms, eliminating the need for repeated data
    loading.
    """

    _filename: str
    _test_ratio: float = 0.2
    _holdout_ratio: float | None = 0.0
    _dataset_card_filename: str | None = None

    df: pd.DataFrame = field(init=False)
    df_train: pd.DataFrame = field(init=False)
    df_test: pd.DataFrame = field(init=False)
    df_holdout: pd.DataFrame | None = field(init=False)

    df_train_id: pd.Series = field(init=False)
    df_test_id: pd.Series = field(init=False)
    df_holdout_id: pd.Series | None = field(init=False)

    y: np.ndarray = field(init=False)
    X: np.ndarray = field(init=False)
    X_pred: np.ndarray = field(init=False)

    dataset_card: YAML | None = field(init=False)
    dataset_config: DatasetCardConfig | None = field(init=False)
    datetime_features: list[str] | None = field(init=False, factory=list)
    hash_column: str | None = field(init=False)

    def __attrs_post_init__(self):
        """Post init method."""
        if self._dataset_card_filename:
            self.load_dataset_card(self._dataset_card_filename)
            self.load_dataset_config(self.dataset_card)

        self.load_data()

    @property
    def filename(self) -> str:
        """
        Return the filename.

        Returns
        -------
        str
            The filename.
        """
        return self._filename

    @property
    def base_filename(self) -> str:
        """
        Return the base filename.

        Returns
        -------
        str
            The filename.
        """
        return self._filename.split("/")[-1].split(".")[0]

    def load_dataset_card(self, dataset_card_filename: str | None = None) -> None:
        """
        Load the dataset card.

        Parameters
        ----------
        dataset_card_filename : str
            The dataset card filename.

        Returns
        -------
        None
        """
        if dataset_card_filename or self._dataset_card_filename:
            self.dataset_card = fetch_dataset_card(
                dataset_card_filename=dataset_card_filename or self._dataset_card_filename
            )

    def load_dataset_config(self, dataset_card: YAML | None = None) -> None:
        """
        Load the dataset config.

        Parameters
        ----------
        dataset_card : YAML
            The dataset card.

        Returns
        -------
        None
        """
        if bool(dataset_card.data):
            self.dataset_card = dataset_card

        if bool(self.dataset_card.data):
            self.dataset_config = DatasetCardConfig(**self.dataset_card.data)
            self.datetime_features = self.dataset_config.datetime_features or []

    def load_full_dataset(self) -> pd.DataFrame:
        """
        Load and return the specified dataset.

        Returns
        -------
        pd.DataFrame
            The loaded dataset in the form of a pandas' DataFrame.

        Raises
        ------
        OSError If the specified dataset is not found.
        TypeError If the dataset file type is not supported.
        """
        path = Path(self.filename)
        if not path.is_file():
            raise OSError(f"Dataset {self.filename} not found.")

        if self.filename.endswith(".csv"):
            return pd.read_csv(self.filename, parse_dates=self.datetime_features)

        elif self.filename.endswith(".json"):
            return pd.read_json(self.filename)

        raise TypeError(f"Dataset file type {self.filename.split('.')[-1]} not supported.")

    def load_data(self, filename: str | None = None, hash_column: str | None = "index") -> None:
        """
        Load data from the given file into Pandas DataFrames and NumPy arrays.

        Parameters
        ----------
        filename : Optional[str]
            The path to the file containing the data.

        hash_column : str
            The column to use for hashing.

        Returns
        -------
        None
        """
        self._filename = filename or self.filename
        self.df = self.load_full_dataset()

        assert not self.df.empty, "No data was loaded, cannot proceed further."

        if not self.dataset_config:
            self.hash_column = "index"
        else:
            self.hash_column = self.dataset_config.id_column or hash_column

        if self.hash_column == "index":
            self.df.reset_index(inplace=True)

        # Unpacking the data
        self.unpack()

        if not self.dataset_config:
            columns_to_drop = []
        else:
            columns_to_drop = self.dataset_config.features_to_drop or []

        if "index" in self.df_train:
            columns_to_drop.append("index")

        self.df_train = self.df_train.reset_index(drop=True).drop(columns=columns_to_drop)
        self.df_test = self.df_test.reset_index(drop=True).drop(columns=columns_to_drop)
        if self.df_holdout is not None:
            self.df_holdout = self.df_holdout.reset_index(drop=True).drop(columns=columns_to_drop)

        # Extracting features and labels
        feature_columns = [cols for cols in self.df.columns if cols not in columns_to_drop]
        self.X = self.df_train[feature_columns].values
        self.X_pred = self.df_test[feature_columns].values
        self.y = self.df_train[self.dataset_config.target].values

    def unpack(self):
        """
        Unpack the loaded data into instance variables.

        Returns
        -------
        None
        """
        # Using split_data_with_id_hash to split data
        self.df_train, self.df_test, self.df_holdout = split_data_with_id_hash(
            self.df, self._test_ratio, self.hash_column, self._holdout_ratio
        )

        # Extracting IDs
        self.df_train_id = self.df_train[self.hash_column]
        self.df_test_id = self.df_test[self.hash_column]
        self.df_holdout_id = (
            self.df_holdout[self.hash_column] if self.df_holdout is not None else None
        )
        # self.all_features = self.df_train.columns.unique()

    def update_data(self, features: list[str]):
        """
        Update the data frames to only include the most important features.

        Parameters
        ----------
        features : List[str]
            The list of important feature names.

        Returns
        -------
        None
        """
        self.df_train = self.df_train[features]
        self.df_test = self.df_test[features]
        self.X_pred = self.df_test.values

    def save_data(
        self,
        base_filename: str | None = None,
        directory: str | None = f"{settings.PROJECT_DIRECTORY}/{settings.DATASET_DIRECTORY}",
        file_type: str | None = "json",
    ) -> None:
        """
        Save the data to the specified directory.

        Parameters
        ----------
        base_filename : str
            The base filename to save the data to.
        directory : str
            The directory to save the data to.
        file_type : str
            The file type to save the data as.
            Default to JSON.

        Returns
        -------
        None
        """
        if not base_filename:
            base_filename = self.base_filename

        if file_type not in {"json", "csv"}:
            raise TypeError(f"File type {file_type} not supported.")

        full_base_filename = f"{directory}/{base_filename}"
        if file_type == "csv":
            self.df_train.to_csv(f"{full_base_filename}_train.csv")
            self.df_test.to_csv(f"{full_base_filename}_test.csv")

            if self.df_holdout is not None:
                self.df_holdout.to_csv(f"{full_base_filename}_holdout.csv")

        self.df_train.to_json(f"{full_base_filename}_train.{file_type}")
        self.df_test.to_json(f"{full_base_filename}_test.{file_type}")

        if self.df_holdout is not None:
            self.df_holdout.to_json(f"{full_base_filename}_holdout.{file_type}")


if __name__ == "__main__":
    # from transaction_risk_profiler.io import DataTransporter

    dt = DataTransporter(
        filename=f"{settings.PROJECT_DIRECTORY}/data/transactions.json",
        test_ratio=0.2,
        holdout_ratio=0.1,
        dataset_card_filename="dataset_card.yml",
    )
    dt.save_data(file_type="csv")
