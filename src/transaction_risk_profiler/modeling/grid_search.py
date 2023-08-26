import logging

import graphlab as gl
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

from transaction_risk_profiler.modeling.baseline import get_probs

logger = logging.getLogger(__name__)


def run_grid_search(Xm, ym):
    df = pd.read_pickle("data/df.pickle")
    bayes_mod = get_probs(BernoulliNB(alpha=0.01), Xm, ym)
    df["bayes_mod"] = bayes_mod

    included_cols = [
        "body_length",
        "channels",
        "fb_published",
        "has_analytics",
        "has_logo",
        "name_length",
        "num_order",
        "num_payouts",
        "object_id",
        "sale_duration2",
        "show_map",
        "user_age",
        "payout_type_n",
        "currency_n",
        "user_type",
        "bayes_mod",
    ]

    y = df["fraud"].values
    X = df[included_cols].values
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    param_grid = {
        "learning_rate": [0.1, 0.15, 0.125],
        "max_depth": [8, 10, 12, 15],
        "min_samples_leaf": [3, 5, 9, 17],
    }
    scorer = make_scorer(metrics.recall_score)
    gbc = GradientBoostingClassifier(n_estimators=2000, max_depth=3)
    clf = GridSearchCV(gbc, param_grid, scoring=scorer)
    clf.fit(X_train, y_train)
    logger.info("params:", clf.best_params_)
    logger.info("recall:", clf.best_score_)


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


def cust_roc(model, train, test):
    return {"train_auc": roc_score(model, train), "test_auc": roc_score(model, test)}


if __name__ == "__main__":
    sf = gl.load_sframe("final.sf")
    train, test = sf.random_split(0.8, seed=25)
    response = "acct_type"
    params = {
        "target": [response],
        "class_weights": "auto",
        "min_child_weight": [0.1, 3, 5, 8],
        "min_loss_reduction": [0.1, 3, 5],
        "max_iterations": [20, 50],
        "step_size": [0.1, 0.3],
    }

    job = gl.grid_search.create(
        (train, test), gl.boosted_trees_classifier.create, params, evaluator=cust_eval
    )
    gl.deploy.jobs.show()

    bt_params = job.get_best_params("test_fn_rate")

    rand_forest_params = {
        "target": response,
        "min_child_weight": [0.1, 0.5, 5, 8, 10],
        "max_depth": [1, 3, 6, 10],
        "max_iterations": [50, 100, 200, 500],
    }

    job = gl.grid_search.create(
        (train, test), gl.random_forest_classifier.create, rand_forest_params, evaluator=cust_eval
    )
    gl.deploy.jobs.show()

    results = job.get_results()[
        ["model_id", "min_child_weight", "max_iterations", "max_depth", "test_fn_rate"]
    ]
    results.sort("test_fn_rate", ascending=True)

    rf_params = job.get_best_params("test_fn_rate")

    knn_search_params = {}

    job = gl.grid_search.create(
        (train, test), gl.nearest_neighbor_classifier.create, knn_search_params, evaluator=cust_eval
    )
    gl.deploy.jobs.show()

    knn_best_params = job.get_best_params("test_fn_rate")
