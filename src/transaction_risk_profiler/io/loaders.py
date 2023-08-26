""" Data and Model Loaders """
import hashlib
import logging
import pickle
from collections import OrderedDict
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterable
from importlib import import_module
from os import path
from pathlib import Path
from typing import Any
from typing import TypeVar

import pandas as pd
import yaml

from transaction_risk_profiler.modeling.build_model import build_model

T = TypeVar("T")
V = TypeVar("V")

logger = logging.getLogger(__name__)


def load_callable(full_callable_name: str) -> callable:
    """
    Loads a callable (either a function or a class) with a fully qualified name.

    Parameters
    ----------
    full_callable_name : str
        The fully qualified name of the callable
        (e.g., 'module.submodule.callable_name').

    Returns
    -------
    Callable
        The loaded callable (either a function or a class).

    Raises
    ------
    ValueError
        If no callable with the given name exists in the specified module.
    """
    full_callable_name.rfind(".")
    module_name, callable_name = full_callable_name.rsplit(sep=".", maxsplit=1)
    module = import_module(module_name)
    _callable = getattr(module, callable_name, None)
    if _callable is None:
        raise ValueError(f"No callable named {full_callable_name} in sys.path")
    return _callable


def dataset_summary_statistics(dataset_location: str) -> pd.DataFrame:
    """
    Return summary statistics for the raw Boston housing dataset.

    This function loads the dataset and calculates the summary
    statistics including the count, mean, standard deviation, minimum, 25th
    percentile, median, 75th percentile, and maximum for each column. The
    result is returned as a pandas' DataFrame.

    Returns
    -------
    pd.DataFrame
    """
    df = load_full_dataset(dataset_location)
    return df.describe()


def load_full_dataset(dataset_location: str) -> pd.DataFrame:
    """
    Load and return the specified dataset.

    Parameters
    ----------
    dataset_location : str, optional
        The name of the dataset to load. Default is 'boston_housing'.

    Returns
    -------
    pd.DataFrame
        The loaded dataset in the form of a pandas' DataFrame.

    Raises
    ------
    ValueError If the specified dataset is not supported.
    """
    json_path = Path(dataset_location)
    if not json_path.is_file():
        raise ValueError(f"Dataset {dataset_location} not found.")
    return pd.read_json(json_path)


def save_dataframe_to_json(df, file_path, orient="split", **kwargs):
    """
    Save a Pandas DataFrame to a JSON file.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to save.
    file_path : str
        The file path where the JSON file will be saved.
    orient : str, optional
        Indication of expected JSON string format. Default is 'split'.
    **kwargs : dict
        Additional keyword arguments for Pandas to_json function.

    Returns
    -------
    None
    """
    df.to_json(file_path, orient=orient, **kwargs)


def load_model(data_filename="data/transactions.json", model_filename="model.pkl"):
    if path.isfile(model_filename):
        with open(model_filename) as f:
            model = pickle.load(f)
    else:
        model = build_model(data_filename, model_filename)
    return model


def yaml_ordered_load(stream, object_pairs_hook=OrderedDict) -> OrderedDict:
    """Parse a yaml file as an OrderedDict.

    Solution comes from: https://stackoverflow.com/a/21912744
    """

    class OrderedLoader(yaml.Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    return yaml.load(stream, OrderedLoader)


def pickle_obj(o: Any, file_name: str) -> None:
    with open(file_name, "wb") as f:
        pickle.dump(o, f)
    logger.info(f"Dump {file_name}")


def unpickle_obj(file_name: str) -> Any:
    with open(file_name, "rb") as f:
        logger.info(f"Unpickled {file_name}")
        return pickle.load(f)


def obj_sha(o: Any) -> str:
    """Calculated the SHA256 checksum of an objects data (get using the pickle module)"""
    return hashlib.sha256(pickle.dumps(o)).hexdigest()


def group_by(key_selector: Callable[[T], V], seq: Iterable[T]) -> dict[V, list[T]]:
    d = defaultdict(list)
    for i in seq:
        d[key_selector(i)].append(i)
    return d


def parse_key_value_tags(tags: list[str]) -> dict[str, str]:
    splits = [tag.split("=") for tag in tags]

    if any(len(pair) != 2 for pair in splits):
        raise ValueError(f"All tags shall be in key=value form: tags={tags}")

    if any(len(pair) != 2 for pair in splits):
        raise ValueError(f"All tag keys shall be unique: tags={tags}")

    return {split[0]: split[1] for split in splits if split[1] != ""}
