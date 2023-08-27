import logging

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

from transaction_risk_profiler.modeling.models.baseline import get_probs

logger = logging.getLogger(__name__)


def run_grid_search(Xm: pd.DataFrame, ym: pd.Series) -> None:
    """
    Run a grid search to optimize the parameters of a Gradient Boosting Classifier.

    Parameters
    ----------
    Xm : pd.DataFrame
        The input features for the model.
    ym : pd.Series
        The target variable for the model.
    """
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
    logger.info(f"Best parameters from grid search: {clf.best_params_}")
    logger.info(f"Best recall score from grid search: {clf.best_score_}")


def load_data() -> tuple:
    """
    Load data and split it into training and testing sets.

    Returns
    -------
    tuple
        Training and testing data.
    """
    df = pd.read_pickle("data/df.pickle")
    train, test = train_test_split(df, test_size=0.2, random_state=25)
    return train, test


def run_boosted_trees_classifier(train: pd.DataFrame, test: pd.DataFrame, response: str) -> dict:
    """
    Run a grid search to optimize the parameters of a Boosted Trees Classifier.

    Parameters
    ----------
    train : pd.DataFrame
        The training data.
    test : pd.DataFrame
        The testing data.
    response : str
        The response variable.

    Returns
    -------
    dict
        The best parameters from the grid search.
    """
    params = {
        "learning_rate": [0.1, 0.15, 0.125],
        "max_depth": [8, 10, 12, 15],
        "min_samples_leaf": [3, 5, 9, 17],
    }
    gbc = GradientBoostingClassifier(n_estimators=2000, max_depth=3)
    clf = GridSearchCV(gbc, params, scoring=make_scorer(metrics.recall_score))
    clf.fit(train, test)
    logger.info("Grid search for boosted trees classifier started.")

    return clf.best_params_


def run_random_forest_classifier(train: pd.DataFrame, test: pd.DataFrame, response: str) -> dict:
    """
    Run a grid search to optimize the parameters of a Random Forest Classifier.

    Parameters
    ----------
    train : pd.DataFrame
        The training data.
    test : pd.DataFrame
        The testing data.
    response : str
        The response variable.

    Returns
    -------
    dict
        The best parameters from the grid search.
    """
    rand_forest_params = {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [1, 3, 6, 10],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
    }
    rfc = RandomForestClassifier()
    clf = GridSearchCV(rfc, rand_forest_params, scoring=make_scorer(metrics.recall_score))
    clf.fit(train, test)
    logger.info("Grid search for random forest classifier started.")

    return clf.best_params_


def run_nearest_neighbor_classifier(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Run a grid search to optimize the parameters of a Nearest Neighbor Classifier.

    Parameters
    ----------
    train : pd.DataFrame
        The training data.
    test : pd.DataFrame
        The testing data.

    Returns
    -------
    dict
        The best parameters from the grid search.
    """
    knn_search_params = {
        "n_neighbors": list(range(1, 31)),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, knn_search_params, scoring=make_scorer(metrics.recall_score))
    clf.fit(train, test)
    logger.info("Grid search for nearest neighbor classifier started.")

    return clf.best_params_


if __name__ == "__main__":
    train, test = load_data()
    response = "acct_type"

    bt_params = run_boosted_trees_classifier(train, test, response)
    rf_params = run_random_forest_classifier(train, test, response)
    knn_best_params = run_nearest_neighbor_classifier(train, test)
