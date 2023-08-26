import os

import numpy as np
from matplotlib import pyplot as plt


def partial_dependency_plots(model, col, data_frame, folder):
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
        ax.set_title(f"partial_dependency_plots of {col}")
        ax.set_ylabel("Fraud")
        ax.set_xlabel(col)
    else:
        if vals.size > 20:
            vals = np.linspace(vals.min(), vals.max(), num=20)
        for val in vals:
            temp[col] = np.repeat(val, n)
            x.append(val)
            y.append(model.predict(temp).mean())
        plt.scatter(x, y)
        plt.title(f"partial_dependency_plots of {col}")
        plt.ylabel("Fraud")
        plt.xlabel(col)

    if not os.path.exists(f"plots/{folder}"):
        os.makedirs(f"plots/{folder}")
    plt.savefig(f"plots/{folder}/{col}.png")
    plt.close()
