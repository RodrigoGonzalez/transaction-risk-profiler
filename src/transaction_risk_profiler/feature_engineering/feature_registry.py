import os
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from typing import Any
from typing import SupportsInt

from feature_engineering.features import FEATURE_MAP_FILE_NAME
from feature_engineering.utils.enums import DataTypeEnum
from feature_engineering.utils.enums import FeatureFamilyEnum
from feature_engineering.utils.enums import FeatureSpecsKeys
from feature_engineering.utils.enums import FeatureTypeEnum
from feature_engineering.utils.enums import MatViewTableNameEnum
from feature_engineering.utils.enums import ProcessingEnum
from feature_engineering.utils.enums import TableNameEnum

from transaction_risk_profiler.io.loaders import load_callable
from transaction_risk_profiler.io.loaders import yaml_ordered_load


@dataclass
class FeatureSpecs:
    """
    A class containing metadata for a single feature record defined in
    feature_registry_metadata
    """

    name: str
    column_name: str
    label: str
    version: SupportsInt
    date_added: date
    description: str
    feature_type: FeatureTypeEnum
    data_type: DataTypeEnum
    feature_family: FeatureFamilyEnum
    kia: bool
    dnd: bool
    function: str
    source_table: TableNameEnum | None
    matvw_table: MatViewTableNameEnum | None
    raw_table: TableNameEnum | None
    pre_processing: list[ProcessingEnum] | None
    post_processing: list[ProcessingEnum] | None
    parent_features: list[str] | None
    save_table: TableNameEnum | None
    contact: str | None
    url: list[str] | None


class FeatureRegistry:
    """
    A service used to read, parse, and validate the feature registry metadata used
    to define all of the features.
    """

    data: dict = {}
    default_feature_registry_path = FEATURE_MAP_FILE_NAME

    def __init__(self, feature_registry_path: str | None = None):
        self.feature_registry = None

        # save location of feature metadata file
        self._feature_registry_path = (
            feature_registry_path or FeatureRegistry.get_default_feature_registry_file_path()
        )
        self._raw_feature_registry_metadata = self._load_raw_feature_registry_metadata()
        self._feature_registry_metadata = self._construct_raw_feature_registry_metadata(
            self._raw_feature_registry_metadata
        )
        self.data = self._construct_feature_registry(self._feature_registry_metadata)

    def get(self, key: str, default: str | None) -> FeatureSpecs:
        """
        Dict-like get method to feature_registry by task key

        Parameters
        ----------
        key : Text
            The column name
        default : Text
            The default value to return if the key does not exist

        Returns
        -------
        feature_metadata : FeatureSpecs
            The current feature metadata record
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        return self.data.get(key, default)

    def __iter__(self) -> Iterator[str]:
        """
        Iterate through column name keys

        Returns
        -------
        iterator : Generator
            An iterator that returns the name of each key
        """
        return iter(self.data.keys())

    def __getitem__(self, key: str) -> Any | None:
        """
        Dict-like access to feature_registry values by column_name key

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        feature_metadata : FeatureSpecs
            The metadata record stored for each feature in the feature registry
        """
        return self.data.get(key)

    def keys(self) -> list[str]:
        """
        Return the list of keys in the data

        Returns
        -------
        keys : list
            The column names of every feature
        """
        return list(self.data.keys())

    def values(self) -> list[FeatureSpecs]:
        """
        Return the list of values in the data

        Returns
        -------
        list_feature_metadata : list
            A list of FeatureSpecs classes for every feature
        """
        return list(self.data.values())

    def items(self) -> list[tuple]:
        """
        Return the list of values in the data

        Returns
        -------
        items : list
            A list of the items in the feature registry
        """
        return list(self.data.items())

    @staticmethod
    def get_default_feature_registry_file_path() -> str:
        """
        Get the default path for the feature registry metadata yaml

        Returns
        -------
        file_path : Text
            The location of the feature registry yaml file
        """
        __file__ = "feature_engineering/features/feature_registry_service.py"
        current_dir = os.path.dirname(__file__)
        return os.path.abspath(
            os.path.join(current_dir, FeatureRegistry.default_feature_registry_path)
        )

    def _load_raw_feature_registry_metadata(self) -> dict:
        """
        Load the yaml file containing the feature metadata

        Returns
        -------
        raw_feature_registry_metadata : dict
            Dictionary containing the contents of the feature registry metadata yaml file
        """
        return yaml_ordered_load(open(self._feature_registry_path, "rb"))

    def _construct_raw_feature_registry_metadata(
        self, raw_feature_registry_metadata: dict | None
    ) -> dict:
        """
        Construct a dictionary, where a column name yields a list of every feature record.

        Parameters
        ----------
        raw_feature_registry_metadata : dict
            Dictionary containing the contents of the feature registry metadata yaml file

        Returns
        -------
        feature_registry_metadata : dict
            Dictionary containing every record for a given feature
        """
        raw_feature_registry_metadata = (
            raw_feature_registry_metadata or self._raw_feature_registry_metadata
        )
        feature_registry_metadata = defaultdict(list)

        for feature_record in raw_feature_registry_metadata[FeatureSpecsKeys.FEATURES.value]:
            meta_date = FeatureSpecs(**feature_record)
            feature_registry_metadata[meta_date.column_name].append(meta_date)

        return feature_registry_metadata

    def _construct_feature_registry(self, feature_registry_metadata: dict | None) -> dict:
        """
        Construct a dictionary, containing the feature registry

        Parameters
        ----------
        feature_registry_metadata : dict
            Dictionary containing every record for a given feature

        Returns
        -------
        dict
        """
        feature_registry_metadata = feature_registry_metadata or self._feature_registry_metadata
        data = {}

        for key, feature_metadata in feature_registry_metadata.items():
            metadata_dict = {record.version: record for record in feature_metadata}
            data[key] = metadata_dict[max(metadata_dict.keys())]

        return data

    def get_feature_history(self, key: str) -> list[str]:
        """
        Feature history

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        feature_history
        """
        return [3 * key]

    def get_feature_name(self, key: str) -> str:
        """
        Feature name used to store the feature record. May or may not be the
        same as the column name

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        name : Text
            The name of the feature used to store the feature
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.name

    def get_feature_label(self, key: str) -> str:
        """
        Gets the readable label of the feature

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        feature_label : Text
            A descriptive/readable label of the feature
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.label

    def get_feature_version(self, key: str) -> SupportsInt:
        """
        The version number of the latest feature spec to use when making/saving/loading
        the feature

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        version : int
            The version of the feature in use for that specific column name
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.version

    def get_feature_date_added(self, key: str) -> date:
        """
        The date the current version of the feature is added to the feature
        ngineering pipeline

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        date_added : datetime.date
            The date that the feature was added to the pipeline or updated
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.date_added

    def get_feature_description(self, key: str) -> str:
        """
        A description of the feature

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        description : Text
            A string containing a description of the feature
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.description

    def get_feature_type(self, key: str) -> FeatureTypeEnum:
        """
        The feature type. Tells us whether the feature is loaded at the start
        of the feature engineering pipeline or derived in the feature
        engineering pipeline.

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        feature_type : Text
            Whether the feature is loaded or derived in the feature engineering pipeline
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.feature_type

    def get_feature_data_type(self, key: str) -> DataTypeEnum:
        """
        The sql data type used when storing the feature. The options are in DataTypeEnum

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        data_type : Text
            The sql data type
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.data_type

    def get_feature_family(self, key: str) -> FeatureFamilyEnum:
        """
        Features are divided into different families, which provide information on
        how they are made. This is the family the feature belongs to as defined
        in FeatureFamilyEnum

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        feature_family : Text
            The name of the feature family
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.feature_family

    def get_feature_kia(self, key: str) -> bool:
        """
        For time series projects define if feature is kia. Sets `known_in_advance`
        within the FeatureSettings class in `dr.partitioning_methods`

        Sets whether the feature is known in advance, i.e., values for future
        dates are known at prediction time. If not specified, the feature uses
        the value from the `default_to_known_in_advance` flag.

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        kia : Text
            Whether the feature should be recognized as known in advance within DataRobot
            dr.FeatureSettings in partitioning_methods
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.kia

    def get_feature_dnd(self, key: str) -> bool:
        """
        For time series projects define if feature is dnd. Sets `do_not_derive`
        within the FeatureSettings class in `dr.partitioning_methods`.

        Sets whether the feature is to be excluded from feature derivation in the
        DataRobot project. If not specified, the feature uses the value from the
        `default_to_do_not_derive` flag.

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        dnd : bool
            Whether to not derive the feature within DataRobot dr.FeatureSettings in
            partitioning_methods
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.dnd

    def get_feature_function(self, key: str) -> Any:
        """
        The function to apply to the data frame to make the feature

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        func : Any
            The function that generates the feature
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return load_callable(metadata.function)

    def get_feature_function_name(self, key: str) -> str:
        """
        The string containing the full function name for the function or metghod that is used to
        generate the feature.

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        feature_function_name : Text
            The name of the function used to derive or load the feature
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.function

    def get_feature_source_table(self, key: str) -> TableNameEnum | None:
        """
        The name of the non materialized view table which contains the stored data from
        which the materialized view is made. The source table for the materialized view.

        DO NOT LOAD DATA USING THIS TABLE NAME!

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        source_table : Text
            The name of a table
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.source_table

    def get_feature_matvw_table(self, key: str) -> MatViewTableNameEnum | None:
        """
        The name of the table to load data from for that feature if the feature is not made in
        the feature engineering pipeline

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        matvw_table : Text
            The name of a materialized view table
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.matvw_table

    def get_feature_raw_table(self, key: str) -> TableNameEnum | None:
        """
        The name of the table where the raw, untransformed, unprocessed data is stored.

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        raw_table : Text
            The name of a table
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.raw_table

    def get_feature_pre_processing(self, key: str) -> list[ProcessingEnum] | None:
        """
        The list of preprocessing done on features that are loaded from a table in the feature
        engineering pipeline

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        pre_processing : list
            A list of the preprocessing that has already been applied to a feature
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.pre_processing

    def get_feature_post_processing(self, key: str) -> list[ProcessingEnum] | None:
        """
        Get the list of postprocessing transformations to apply after the features
        have all been made.

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        post_processing : list
            A list of postprocessing steps to apply after the feature has been made
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.post_processing

    def get_feature_parent_features(self, key: str) -> list[str] | None:
        """
        The names of the features that are necessary to derive prior to deriving the named
        feature. The parent features.

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        parent_features : list
            A list of the feature column names
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.parent_features

    def get_feature_save_table(self, key: str) -> TableNameEnum | None:
        """
        The name of a table to save the feature to after the feature engineering pipeline has run

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        save_table : Text
            The name of the table to save the feature to
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.save_table

    def get_feature_contact(self, key: str) -> str | None:
        """
        The name of a person that is knowledgeable about the feature, or the name of the person
        that added it.

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        contact : Text
            Person's name
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.contact

    def get_feature_url(self, key: str) -> list[str] | None:
        """
        Get a list of url's containing information about the feature

        Parameters
        ----------
        key : Text
            The column name of the feature property to return

        Returns
        -------
        urls : list
            A list of url's
        """
        assert key in self.keys(), f"key: <{key}> not in feature registry"
        metadata = self.get(key, None)
        return metadata.url


if __name__ == "__main__":
    feature_registry = FeatureRegistry()
