""" Module for creating bar charts for EDA. """
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_comparative_barchart(
    series1: pd.Series,
    series2: pd.Series,
    column_label: str,
    bar_width: float = 0.5,
    title: str = "Comparative Bar Chart",
    x_label: str = "X-axis",
    y_label: str = "Y-axis",
) -> None:
    """
    Plots a comparative bar chart for two Pandas' Series.

    Parameters
    ----------
    series1 : pd.Series
        The first pandas Series for comparison.
    series2 : pd.Series
        The second pandas Series for comparison.
    column_label : str
        The label for the column.
    bar_width : float, optional
        The width of the bars in the chart, by default 0.5.
    title : str, optional
        The title of the chart, by default 'Comparative Bar Chart'.
    x_label : str, optional
        The label for the x-axis, by default 'X-axis'.
    y_label : str, optional
        The label for the y-axis, by default 'Y-axis'.
    """
    series1_counts = series1.value_counts(dropna=True)
    series2_counts = series2.value_counts(dropna=True)
    ticks = list(set(series1_counts.keys()).union(series2_counts.keys()))
    values1 = [series1_counts.get(tick, 0) for tick in ticks]
    values2 = [series2_counts.get(tick, 0) for tick in ticks]
    values1_norm = values1 / np.sum(values1)
    values2_norm = values2 / np.sum(values2)
    indices = np.arange(len(ticks))

    fig, ax = plt.subplots()
    ax.bar(indices, values1_norm, bar_width, color="blue", alpha=0.5, label=series1.name)
    ax.bar(indices, values2_norm, bar_width, color="green", alpha=0.5, label=series2.name)
    ax.set_xticks(indices + bar_width / 2.0)
    ax.set_xticklabels(ticks)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc="best")
    plt.grid()
    plt.show()


def plot_comparative_barchart_value_counts(
    series1_counts: pd.Series,
    series2_counts: pd.Series,
    column_label: str,
    bar_width: float = 0.5,
    title: str = "Comparative Bar Chart",
    x_label: str = "X-axis",
    y_label: str = "Y-axis",
) -> None:
    """
    Plots a comparative bar chart for two Pandas' Series using precomputed value counts.

    Parameters
    ----------
    series1_counts : pd.Series
        The value counts of the first pandas Series for comparison.
    series2_counts : pd.Series
        The value counts of the second pandas Series for comparison.
    column_label : str
        The label for the column.
    bar_width : float, optional
        The width of the bars in the chart, by default 0.5.
    title : str, optional
        The title of the chart, by default 'Comparative Bar Chart'.
    x_label : str, optional
        The label for the x-axis, by default 'X-axis'.
    y_label : str, optional
        The label for the y-axis, by default 'Y-axis'.
    """
    ticks = list(set(series1_counts.keys()).union(series2_counts.keys()))
    values1 = [series1_counts.get(tick, 0) for tick in ticks]
    values2 = [series2_counts.get(tick, 0) for tick in ticks]
    values1_norm = values1 / np.sum(values1)
    values2_norm = values2 / np.sum(values2)
    indices = np.arange(len(ticks))

    fig, ax = plt.subplots()
    ax.bar(indices, values1_norm, bar_width, color="blue", alpha=0.5, label=series1_counts.name)
    ax.bar(indices, values2_norm, bar_width, color="green", alpha=0.5, label=series2_counts.name)
    ax.set_xticks(indices + bar_width / 2.0)
    ax.set_xticklabels(ticks)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc="best")
    plt.grid()
    plt.show()
