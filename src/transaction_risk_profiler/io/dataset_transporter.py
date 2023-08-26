from pathlib import Path

import numpy as np
import pandas as pd
from attrs import define
from attrs import field

from transaction_risk_profiler.io.loaders import unpickle_obj

# columns_enum = DataColumnsEnum()
# dataset_config = DatasetCardConfig(**dataset_card.data)


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
    all_features: pd.Index = field(init=False)

    def __post_init__(self):
        self.load_data(self.filename)
        self.unpack()

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
            return pd.read_csv(self.filename)
        elif self.filename.endswith(".json"):
            return pd.read_json(self.filename)
        elif self.filename.endswith(".pkl"):
            return unpickle_obj(self.filename)

        raise TypeError(f"Dataset file type {self.filename.split('.')[-1]} not supported.")

    def load_data(self, filename: str | None) -> None:
        """
        Load data from the given CSV file into Pandas DataFrames and NumPy arrays.

        Parameters
        ----------
        filename : str
            The path to the CSV file containing the data.

        Returns
        -------
        None
        """
        print("\nLoading Data")
        self._filename = filename or self.filename
        self.df = self.load_full_dataset()
        self.df_train = self.df.loc[:249, "var_1":"var_300"]
        self.df_test = self.df.loc[250:, "var_1":"var_300"]
        self.df_train_id = self.df.loc[:249, "id"]
        self.df_test_id = self.df.loc[250:, "id"]
        self.y = self.df.loc[:249, "target_eval"].values
        self.X = self.df_train.values
        self.X_pred = self.df_test.values

    def unpack(self):
        """
        Unpack the loaded data into instance variables.

        Returns
        -------
        None
        """
        print("\nLoading Data")
        self.all_features = self.df_train.columns.unique()

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
