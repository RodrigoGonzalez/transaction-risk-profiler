""" Simple feature engineering transforms. """
import pandas as pd


def fill_na_with_value(df: pd.DataFrame, column: str, value) -> None:
    df[column].fillna(value=value, inplace=True)


def mismatch_country(df: pd.DataFrame, new_column: str, column_1: str, column_2: str) -> None:
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

    Returns
    -------
    None
        Updates the DataFrame in place.
    """
    for value in feature_values:
        new_column_name = f"{column_prefix}_{value}"
        df[new_column_name] = df[feature_name] == value


def proportion_non_empty(cells: list[dict], field_name: str = "address") -> float:
    """
    Calculate the proportion of cells with non-empty 'address' fields.

    Parameters
    ----------
    cells : list[dict[str, str]]
        A list of dictionaries, each containing an 'address' field.
    field_name : str, optional
        The field to check for non-empty values. Default is 'address'.



    Returns
    -------
    float
        The proportion of cells with non-empty 'address' fields.
    """
    total_cells = len(cells)

    if total_cells == 0:
        return 0.0

    non_empty_addresses = sum(bool(cell[field_name].strip()) for cell in cells)
    return 1 - (non_empty_addresses / float(total_cells))
