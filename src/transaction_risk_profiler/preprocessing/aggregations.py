from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterable
from copy import deepcopy
from typing import Any
from typing import TypeVar

import pandas as pd

T = TypeVar("T")
V = TypeVar("V")


def group_by_elements(key_selector: Callable[[T], V], sequence: Iterable[T]) -> dict[V, list[T]]:
    """
    Groups elements in an iterable by a given key selector function.

    The function takes an iterable and a key selector function. It iterates
    through the iterable, applying the key selector function to each element.
    The function then groups elements by the unique keys generated by the key
    selector.

    Parameters
    ----------
    key_selector : Callable[[T], V]
        A function that takes an element of type `T` and returns a key of type
        `V`.

    sequence : Iterable[T]
        An iterable of elements of a type `T`.

    Returns
    -------
    dict[V, list[T]]
        A dictionary where each key is of type `V` and maps to a list of
        elements of a type `T`.

    Examples
    --------
    >>> group_by_elements(lambda x: x % 2, [1, 2, 3, 4, 5])
    {1: [1, 3, 5], 0: [2, 4]}
    """
    grouped_elements = defaultdict(list)

    for element in sequence:
        key = key_selector(element)
        grouped_elements[key].append(element)

    return grouped_elements


def create_agg_schema(item: str) -> dict[str, dict[str, Any]]:
    """
    Creates an aggregation schema dictionary for a specific item.

    This function generates a dictionary that maps the type of the field (e.g.,
    'amount', 'country', 'state') to a dictionary of aggregation functions
    (e.g., sum, min, max, mean, etc.). The function also appends the item's
    name to the aggregation keys.

    Parameters
    ----------
    item : str
        The name of the item for which the aggregation schema is created.

    Returns
    -------
    dict[str, dict[str, Any]]
        A nested dictionary containing the aggregation schema.

    Examples
    --------
    >>> create_agg_schema('product')
    {
        'amount': {
            'product_amt_sum': 'sum',
            'product_amt_min': 'min',
            'product_amt_max': 'max',
            'product_amt_mean': 'mean',
            'product_amt_count': 'count',
            'product_amt_std': 'std'
        },
        'country': {
            'product_country_count': <function>
        },
        'state': {
            'product_state_count': <function>
        }
    }
    """
    unique_count: Callable = lambda x: x.nunique()

    return {
        "amount": {
            f"{item}_amt_sum": "sum",
            f"{item}_amt_min": "min",
            f"{item}_amt_max": "max",
            f"{item}_amt_mean": "mean",
            f"{item}_amt_count": "count",
            f"{item}_amt_std": "std",
        },
        "country": {f"{item}_country_count": unique_count},
        "state": {f"{item}_state_count": unique_count},
        "quantity_sold": {"total_sold": "sum"},
        "cost": {
            "min_cost": "min",
            "med_cost": "median",
            "max_cost": "max",
            "mean_cost": "mean",
            "std_cost": "std",
            "options": "count",
        },
        "created": {"min_created": "min", "max_created": "max"},
        "event_id": {"unique_events": unique_count},
    }


def get_payout_info(df):
    """
    Retrieves payout information from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing 'object_id' and 'previous_payouts'
        columns.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing extracted payout information.
    """
    payout_columns = [
        "object_id",
        "address",
        "amount",
        "country",
        "created",
        "event",
        "name",
        "state",
        "uid",
        "zip_code",
    ]

    element_list = [
        (
            df.object_id[idx],
            element["address"],
            element["amount"],
            element["country"],
            element["created"],
            element["event"],
            element["name"],
            element["state"],
            element["uid"],
            element["zip_code"],
        )
        for idx in range(df.shape[0])
        for element in df.previous_payouts[idx]
    ]
    return pd.DataFrame(element_list, columns=payout_columns)


def get_ticket_info(df):
    """
    Retrieves ticket information from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing 'object_id' and 'ticket_types' columns.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing extracted ticket information.
    """
    ticket_columns = [
        "object_id",
        "availability",
        "cost",
        "event_id",
        "quantity_sold",
        "quantity_total",
    ]

    element_list = [
        (
            df.object_id[idx],
            element["availability"],
            element["cost"],
            element["event_id"],
            element["quantity_sold"],
            element["quantity_total"],
        )
        for idx in range(df.shape[0])
        for element in df.ticket_types[idx]
    ]
    return pd.DataFrame(element_list, columns=ticket_columns)


def apply_aggregations(group: pd.DataFrame, flat_agg_func: dict) -> pd.Series:
    """
    Applies a set of aggregation functions to a given DataFrame group.

    This function iterates over each column-function pair in the global
    `flat_agg_func` dictionary.

    For each pair, it checks if the column exists in the group DataFrame.
    If it does, it applies the aggregation function(s) to the column. The
    results are stored in a dictionary where the keys are the
    column-function pairs and the values are the results of the aggregation
    functions.

    Parameters
    ----------
    group : pd.DataFrame
        The DataFrame group to which the aggregation functions are applied.
    flat_agg_func : dict
        A dictionary where the keys are column names and the values are
        aggregation functions applicable to these columns.

    Returns
    -------
    pd.Series
        A series where the index is the column-function pairs and the
        values are the results of the aggregation functions.
    """
    result = {}
    for col, funcs in flat_agg_func.items():
        if col in group.columns:
            for func in funcs:
                result_key = f"{col}_{func.__name__}" if callable(func) else f"{col}_{func}"
                result[result_key] = func(group[col]) if callable(func) else group[col].agg(func)
    return pd.Series(result)


def weighted_cost(x: pd.DataFrame) -> float:
    """
    Computes the weighted cost based on the 'cost' and 'quantity_sold'
    columns of a DataFrame.

    The weighted cost is calculated as the sum of the product of 'cost' and
    'quantity_sold' divided by the total 'quantity_sold'.

    If 'quantity_sold' is not present or its sum is 0, the function returns
    0.

    Parameters
    ----------
    x : pd.DataFrame
        The input DataFrame which must contain the columns "cost" and
        "quantity_sold".

    Returns
    -------
    float
        The computed weighted cost, or 0 if "quantity_sold" is not present
        or its sum is 0.
    """
    return (
        (x["cost"] * x["quantity_sold"]).sum() / x["quantity_sold"].sum()
        if "quantity_sold" in x and x["quantity_sold"].sum() != 0
        else 0
    )


def fraction_sold(x: pd.DataFrame) -> float:
    """
    Computes the fraction of items sold over the total quantity of items.

    The fraction is calculated as the sum of 'quantity_sold' divided by the
    sum of 'quantity_total'.

    If either "quantity_sold" or "quantity_total" is not present in the
    DataFrame or if the sum of "quantity_total" is 0, the function returns
    0.

    Parameters
    ----------
    x : pd.DataFrame
        The input DataFrame which must contain the columns "quantity_sold"
        and "quantity_total".

    Returns
    -------
    float
        The computed fraction of sold items over the total quantity of
        items, or 0 if either "quantity_sold" or "quantity_total" is not
        present or if the sum of "quantity_total" is 0.
    """
    return (
        x["quantity_sold"].sum() / x["quantity_total"].sum()
        if "quantity_sold" in x and "quantity_total" in x and x["quantity_total"].sum() != 0
        else 0
    )


def aggregate_data(
    df: pd.DataFrame, group_col: str, agg_func: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    """
    Aggregates a DataFrame based on a given column and aggregation function.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to aggregate.
    group_col : str
        The name of the column to group the DataFrame by.
    agg_func : dict
        The aggregation function to apply to the grouped data. This should be a
        dictionary where the keys are column names and the values are
        aggregation functions applicable to these columns.

    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame with missing values filled with zeros.

    Example
    -------
    # >>> df = pd.DataFrame({
    # ...     'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
    # ...     'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
    # ...     'C': np.random.randn(8),
    # ...     'D': np.random.randn(8)
    # ... })
    # >>> agg_func = {'C': ['sum', 'min'], 'D': ['max', 'min']}
    # >>> aggregate_data(df, 'A', agg_func)
    """
    df = deepcopy(df)

    # Update your aggregation schema to use these new functions
    agg_func["cost"][f"{group_col}_weighted_cost"] = weighted_cost
    agg_func["quantity_sold"][f"{group_col}_fraction_sold"] = fraction_sold

    # Flatten the nested aggregation dictionary to match Pandas' expected format
    flat_agg_func = {col: list(funcs.values()) for col, funcs in agg_func.items()}

    # Check if the required columns exist in the DataFrame
    required_columns = set(flat_agg_func.keys())
    existing_columns = set(df.columns)

    if not required_columns.issubset(existing_columns):
        missing_columns = required_columns - existing_columns
        print(
            f"Warning: Column(s) {list(missing_columns)} do not exist in the "
            f"DataFrame. Skipping these columns."
        )
        # Remove missing columns from the aggregation schema
        flat_agg_func = {
            col: funcs for col, funcs in flat_agg_func.items() if col in existing_columns
        }

    # Perform the aggregation
    agg_df = df.groupby(group_col).apply(apply_aggregations)

    # Rename columns that use lambda functions
    agg_df.rename(columns=lambda x: x.replace("<lambda>", "unique_count"), inplace=True)
    return agg_df.fillna(0)
