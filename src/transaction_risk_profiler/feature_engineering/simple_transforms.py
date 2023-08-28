"""Module for simple feature engineering transforms.

This module provides functions for performing simple transformations on pandas
DataFrames. These transformations include filling NA values, creating mismatched
country columns, creating feature columns, and calculating the proportion of
non-empty cells.
"""
from typing import Any

import numpy as np
import pandas as pd


def fill_na_with_value(df: pd.DataFrame, column: str, value: int | str) -> None:
    """Fill NA/NaN values in a specified column of a DataFrame with a given value.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to update.
    column : str
        The name of the column to fill NA/NaN values in.
    value : Union[int, str]
        The value to fill NA/NaN values with.
    """
    df[column].fillna(value=value, inplace=True)


def mismatch_country(df: pd.DataFrame, new_column: str, column_1: str, column_2: str) -> None:
    """
    Create a new column in the DataFrame indicating whether the values in
    two other columns match.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to update.
    new_column : str
        The name of the new column to create.
    column_1 : str
        The name of the first column to compare.
    column_2 : str
        The name of the second column to compare.
    """
    df[new_column] = df[column_1] != df[column_2]


def create_feature_columns(
    df: pd.DataFrame, feature_values: list[int | str], column_prefix: str, feature_name: str
) -> None:
    """
    Create new feature columns in the DataFrame based on a list of feature
    values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to update.
    feature_values : List[Union[int, str]]
        The list of feature values to create new columns for.
    column_prefix : str
        The prefix to use for the new column names.
    feature_name : str
        The name of the feature column to process.
    """
    for value in feature_values:
        new_column_name = f"{column_prefix}_{value}"
        df[new_column_name] = df[feature_name] == value


def proportion_non_empty(cells: list[dict], field_name: str = "address") -> float:
    """Calculate the proportion of cells with non-empty fields.

    Parameters
    ----------
    cells : List[dict]
        A list of dictionaries, each containing an 'address' field.
    field_name : str, optional
        The field to check for non-empty values. Default is 'address'.

    Returns
    -------
    float
        The proportion of cells with non-empty fields.
    """
    total_cells = len(cells)

    if total_cells == 0:
        return 0.0

    non_empty_addresses = sum(bool(cell[field_name].strip()) for cell in cells)
    return 1 - (non_empty_addresses / float(total_cells))


def convert_float_to_int(x: float | Any) -> int | Any:
    """
    Convert float to integer, while keeping NaNs unchanged.

    Parameters
    ----------
    x : float or any
        The value to be converted to integer if column value contains a float.

    Returns
    -------
    int or same type as input
        The integer representation of the float if the input contains a float
        and not NaN, otherwise returns the input without modification.
    """
    if isinstance(x, float):
        return x if np.isnan(x) else int(x)
    else:
        return x


def add_mismatch_column(
    df: pd.DataFrame, main_column: str, secondary_column: str, new_col_name: str
) -> pd.DataFrame:
    """
    Add a column to Pandas' DataFrame indicating whether a main_column and
    secondary_column contain different values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to modify.
    main_column : str
        The name of the column containing main column information.
    secondary_column : str
        The name of the column containing comparison column information.
    new_col_name : str
        The name of the new column to add.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with the new column.
    """
    df[new_col_name] = df[main_column] != df[secondary_column]
    return df


def add_is_specific_value(df: pd.DataFrame, main_column: str, column_values: list) -> pd.DataFrame:
    """
    Add columns to DataFrame indicating whether the country matches specific
    column values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to modify.
    main_column : str
        The name of the column containing main column information.
    column_values : list
        A list of values to check for.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with the new columns.
    """
    for column_value in column_values:
        new_col_name = f"is_{column_value.lower()}"
        df[new_col_name] = df[main_column] == column_value
    return df


def add_is_blank_column(df: pd.DataFrame, main_column: str, new_col_name: str) -> pd.DataFrame:
    """
    Add a column to Pandas' DataFrame indicating whether the column entry is
    blank.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to modify.
    main_column : str
        The name of the column containing column value information.
    new_col_name : str
        The name of the new column to add.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with the new column.
    """
    df[new_col_name] = df[main_column] == ""
    return df
