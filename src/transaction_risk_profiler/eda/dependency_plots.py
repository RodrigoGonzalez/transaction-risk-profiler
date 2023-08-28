import os
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def partial_dependency_plots(
    model: Any,
    col: str,
    data_frame: pd.DataFrame,
    folder: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """
    Generates partial dependency plots for a given model and column in a data
    frame, and saves the plots in a specified folder.

    Parameters
    ----------
    model : Any
        The machine learning model that you want to use for making predictions.
        It could be any model that has a predict method, such as a regression
        model or a classification model.
    col : str
        The name of the column in the data_frame that you want to create
        partial dependency plots for.
    data_frame : pd.DataFrame
        The pandas DataFrame that contains the data you want to use for
        creating the partial dependency plots.
    folder : str
        The name of the folder where the plots will be saved.
    title : str
        The title of the plot.
    x_label : str
        The label for the x-axis.
    y_label : str
        The label for the y-axis.
    """
    vals = data_frame[col].unique()
    n = data_frame.shape[0]
    temp = data_frame.copy()
    x, y = [], []
    if isinstance(vals[0], str):
        if len(vals) > 20:
            vals = []
        for val in vals:
            temp[col] = np.repeat(val, n)
            x.append(val)
            y.append(model.predict(temp).mean())
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(y)), y)
        ax.set_xticks(np.arange(len(y)) + 0.35)
        ax.set_xticklabels(x)
        ax.set_title(title)
        ax.set_y_label(y_label)
        ax.set_x_label(x_label)
    else:
        if vals.size > 20:
            vals = np.linspace(vals.min(), vals.max(), num=20)
        for val in vals:
            temp[col] = np.repeat(val, n)
            x.append(val)
            y.append(model.predict(temp).mean())
        plt.scatter(x, y)
        plt.title(title)
        plt.y_label(y_label)
        plt.x_label(x_label)

    if not os.path.exists(f"artifacts/{folder}"):
        os.makedirs(f"artifacts/{folder}")
    plt.savefig(f"artifacts/{folder}/{col}.png")
    plt.close()
