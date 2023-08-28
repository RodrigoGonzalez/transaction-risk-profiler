""" This module contains functions to plot histograms of the data. """
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def comparative_histogram(
    series1: pd.Series,
    series2: pd.Series,
    col_name: str,
    max_val,
    label1: str,
    label2: str,
    n_bins=100,
) -> None:
    """
    Plots comparative histograms of two Pandas' Series.

    Parameters
    ----------
    series1 : pd.Series
        First series to be plotted.
    series2 : pd.Series
        Second series to be plotted.
    col_name : str
        Name of the column to be used as the x-axis label.
    max_val : float
        Maximum value to be used for the x-axis.
    label1 : str
        Label for the first series.
    label2 : str
        Label for the second series.
    n_bins : int, optional
        The number of bins to be used in the histogram, by default 100.

    Returns
    -------
    None
    """
    min1, max1 = min(series1), max(series1)
    min2, max2 = min(series2), max(series2)
    min_val, max_val = min(min1, min2), min(max(max1, max2), max_val)
    bins = np.linspace(min_val, max_val, n_bins)
    plt.hist(series1, bins, alpha=0.5, density=True, label=label1)
    plt.hist(series2, bins, alpha=0.5, density=True, label=label2)
    plt.legend(loc="best")
    plt.xlabel(col_name)
    plt.ylabel("normalized frequency")
    plt.title(col_name)
    plt.grid()
    plt.show()
