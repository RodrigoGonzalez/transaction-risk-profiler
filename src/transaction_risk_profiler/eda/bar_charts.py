""" Module for creating bar charts for EDA. """
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def comparative_barchart(
    col_df_fraud: pd.Series, col_df_premium: pd.Series, col_name: str, bar_chart_width: float = 0.5
) -> None:
    fraud_value_counts = col_df_fraud.value_counts(dropna=True)
    premium_value_counts = col_df_premium.value_counts(dropna=True)
    x_ticks_f = list(fraud_value_counts.keys())
    x_ticks_p = list(premium_value_counts.keys())
    x_ticks = list(set(x_ticks_f + x_ticks_p))
    val_f: list[float] = []
    val_p: list[float] = []
    for x_tick in x_ticks:
        if x_tick in fraud_value_counts.keys():
            val_f.append(fraud_value_counts[x_tick])
        else:
            val_f.append(0)
        if x_tick in premium_value_counts.keys():
            val_p.append(premium_value_counts[x_tick])
        else:
            val_p.append(0)
    val_f_norm = val_f / (sum(val_f) * 1.0)
    val_p_norm = val_p / (sum(val_p) * 1.0)
    # now that ticks and values lists created, can make plot
    n_ticks = len(x_ticks)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(n_ticks)
    ax.bar(ind, val_f_norm, bar_chart_width, color="blue", alpha=0.5, label="fraud")
    ax.bar(ind, val_p_norm, bar_chart_width, color="green", alpha=0.5, label="premium")
    x_tick_marks = [str(x_tick) for x_tick in x_ticks]
    ax.set_x_ticks(ind + bar_chart_width / 2.0)
    ax.set_x_ticklabels(x_tick_marks)
    ax.set_ylabel("normalized frequency")
    ax.set_xlabel(col_name)
    ax.legend(loc="best")
    plt.grid()
    plt.show()


def comparative_barchart_value_counts(
    fraud_value_counts: pd.Series,
    premium_value_counts: pd.Series,
    col_name: str,
    bar_chart_width: float = 0.5,
) -> None:
    x_ticks_f = list(fraud_value_counts.keys())
    x_ticks_p = list(premium_value_counts.keys())
    x_ticks = list(set(x_ticks_f + x_ticks_p))
    val_f = []
    val_p = []
    for x_tick in x_ticks:
        if x_tick in fraud_value_counts.keys():
            val_f.append(fraud_value_counts[x_tick])
        else:
            val_f.append(0)
        if x_tick in premium_value_counts.keys():
            val_p.append(premium_value_counts[x_tick])
        else:
            val_p.append(0)
    val_f_norm = val_f / (sum(val_f) * 1.0)
    val_p_norm = val_p / (sum(val_p) * 1.0)
    # now that ticks and values lists created, can make plot
    n_ticks = len(x_ticks)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(n_ticks)

    ax.bar(ind, val_f_norm, bar_chart_width, color="blue", alpha=0.5, label="fraud")
    ax.bar(ind, val_p_norm, bar_chart_width, color="green", alpha=0.5, label="premium")
    x_tick_marks = [str(x_tick) for x_tick in x_ticks]
    ax.set_x_ticklabels(x_tick_marks)
    ax.set_x_ticks(ind + bar_chart_width / 2.0)
    ax.set_ylabel("normalized frequency")
    ax.set_xlabel(col_name)
    ax.legend(loc="best")
    plt.grid()
