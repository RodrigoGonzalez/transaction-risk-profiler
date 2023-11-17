""" Data and Model Loaders """
import hashlib
import logging
import pickle
from collections import OrderedDict
from collections.abc import Callable
from importlib import import_module
from os import path
from pathlib import Path
from typing import Any
from typing import TypeVar
from typing import cast

import pandas as pd
from yaml import Loader
from yaml import load
from yaml import resolver

from transaction_risk_profiler.modeling.models.build_model import build_model

T = TypeVar("T")
V = TypeVar("V")

logger = logging.getLogger(__name__)


def load_callable(full_callable_name: str) -> Callable:
    """
    Loads a Callable (either a function or a class) with a fully qualified name.

    Parameters
    ----------
    full_callable_name : str
        The fully qualified name of the Callable
        (e.g., 'module.submodule.callable_name').

    Returns
    -------
    Callable
        The loaded Callable (either a function or a class).

    Raises
    ------
    ValueError
        If no Callable with the given name exists in the specified module.
    """
    full_callable_name.rfind(".")
    module_name, callable_name = full_callable_name.rsplit(sep=".", maxsplit=1)
    module = import_module(module_name)
    _callable = getattr(module, callable_name, None)
    if _callable is None:
        raise ValueError(f"No Callable named {full_callable_name} in sys.path")
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


def construct_ordered_mapping(loader: Loader, node: Any) -> OrderedDict:
    """
    Construct ordered mapping for YAML loader.

    This function transforms a YAML node into an ordered dictionary.

    Parameters
    ----------
    loader : yaml.Loader
        The YAML loader.
    node : Any
        The YAML node to transform.

    Returns
    -------
    OrderedDict
        The ordered dictionary representation of the node.
    """
    loader.flatten_mapping(node)
    return OrderedDict(loader.construct_pairs(node))


def yaml_ordered_load(stream: Any, object_pairs_hook: Callable = OrderedDict) -> OrderedDict:
    """
    Parse a YAML file as an OrderedDict.

    This function uses a custom YAML Loader that employs an OrderedDict to maintain the order
    of items as they appear in the YAML file.

    Parameters
    ----------
    stream : Any
        The YAML file stream.
    object_pairs_hook : Callable, optional
        The ordered dictionary type to use, by default OrderedDict.

    Returns
    -------
    OrderedDict
        The ordered dictionary representation of the YAML file.

    Notes
    -----
    Solution adapted from: https://stackoverflow.com/a/21912744
    """

    if isinstance(stream, Path):
        stream.resolve()
        stream = Path(stream).read_text()
    ordered_loader = cast(type[Loader], type("OrderedLoader", (Loader,), {}))
    ordered_loader.add_constructor(
        resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_ordered_mapping
    )
    return load(stream, ordered_loader)


def pickle_obj(obj: Any, file_name: str) -> None:
    """
    Serialize an object and save it to a file.

    Parameters
    ----------
    obj : Any
        The object to be pickled.
    file_name : str
        The name of the file where the object will be saved.

    Returns
    -------
    None
    """
    try:
        with open(file_name, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object successfully dumped to {file_name}.")

    except FileNotFoundError:
        logger.exception("File not found during pickling.")

    except PermissionError:
        logger.exception("Permission error during pickling.")

    except pickle.PickleError:
        logger.exception("An error occurred while pickling the object.")

    except Exception:
        logger.exception("An unknown error occurred during pickling.")


def unpickle_obj(file_name: str) -> Any:
    """
    Load a serialized object from a file.

    Parameters
    ----------
    file_name : str
        The name of the file from which the object will be loaded.

    Returns
    -------
    Any
        The unpickled object.
    """
    try:
        with open(file_name, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Object successfully loaded from {file_name}.")

    except FileNotFoundError:
        logger.exception("File not found during unpickling.")

    except PermissionError:
        logger.exception("Permission error during unpickling.")

    except pickle.PickleError:
        logger.exception("An error occurred while unpickling the object.")

    except Exception:
        logger.exception("An unknown error occurred during unpickling.")

    return obj


def obj_sha(obj: Any) -> str | None:
    """
    Calculate the SHA256 checksum of an object's serialized data.

    Parameters
    ----------
    obj : Any
        The object for which the SHA256 checksum will be calculated.

    Returns
    -------
    str
        The SHA256 checksum of the object's serialized data.
    """
    try:
        return hashlib.sha256(pickle.dumps(obj)).hexdigest()

    except pickle.PickleError:
        logger.exception("An error occurred while serializing the object for checksum calculation.")

    except Exception:
        logger.exception("An unknown error occurred while calculating the SHA256 checksum.")

    return None
