""""Custom evaluation metrics for risk detection models."""
import graphlab as gl
import pandas as pd
from sklearn.metrics import roc_auc_score


def custom_eval(model: gl.boosted_trees_classifier, data: pd.DataFrame) -> float:
    target = model.get("target")
    predictions = model.predict(data, output_type="class")
    misses = 1 - (predictions == data[target])
    return sum(misses[data[target] == "fraud"]) / float(sum(data[target] == "fraud"))


def roc_score(model, data):
    target = model.get("target")
    predictions = model.predict(data, output_type="class")
    return roc_auc_score(data[target], predictions)


def cust_eval(model, train, test):
    return {"train_fn_rate": custom_eval(model, train), "test_fn_rate": custom_eval(model, test)}


def custom_roc(model, train, test):
    return {"train_auc": roc_score(model, train), "test_auc": roc_score(model, test)}
