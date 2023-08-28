""" This module contains functions to plot histograms of the data. """
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def comparative_histogram(
    series_1: pd.Series,
    series_2: pd.Series,
    col_name: str,
    max_val,
    label_1: str,
    label_2: str,
    n_bins=100,
) -> None:
    """
    Plots comparative histograms of two Pandas' Series.

    Parameters
    ----------
    series_1 : pd.Series
        First series to be plotted.
    series_2 : pd.Series
        Second series to be plotted.
    col_name : str
        Name of the column to be used as the x-axis label.
    max_val : float
        Maximum value to be used for the x-axis.
    label_1 : str
        Label for the first series.
    label_2 : str
        Label for the second series.
    n_bins : int, optional
        The number of bins to be used in the histogram, by default 100.

    Returns
    -------
    None
    """
    min_1, max_1 = min(series_1), max(series_1)
    min_2, max_2 = min(series_2), max(series_2)
    min_val, max_val = min(min_1, min_2), min(max(max_1, max_2), max_val)
    bins = np.linspace(min_val, max_val, n_bins)
    plt.hist(series_1, bins, alpha=0.5, density=True, label=label_1)
    plt.hist(series_2, bins, alpha=0.5, density=True, label=label_2)
    plt.legend(loc="best")
    plt.xlabel(col_name)
    plt.ylabel("normalized frequency")
    plt.title(col_name)
    plt.grid()
    plt.show()
