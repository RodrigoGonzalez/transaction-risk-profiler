""" This module contains functions to plot histograms of the data. """
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def comparative_histogram(
    col_df_fraud: pd.Series, col_df_premium: pd.Series, col_name: str, m_x_val, n_bins=100
) -> None:
    fraud = col_df_fraud
    premium = col_df_premium
    f_min, f_max = min(fraud), max(fraud)
    p_min, p_max = min(premium), max(premium)
    m_n, m_x = min(f_min, p_min), max(f_max, p_max)
    m_x = min(m_x, m_x_val)
    # if m_n < 0: m_n = 0
    bins = np.linspace(m_n, m_x, n_bins)
    plt.hist(fraud, bins, alpha=0.5, density=True, label="fraud")
    plt.hist(premium, bins, alpha=0.5, density=True, label="premium")
    plt.legend(loc="best")
    plt.xlabel(col_name)
    plt.ylabel("normalized frequency")
    plt.title(col_name)
    plt.grid()
    plt.show()
