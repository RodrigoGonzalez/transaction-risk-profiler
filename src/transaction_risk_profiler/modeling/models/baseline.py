""" This module generates the baseline models for the transaction risk profiler. """
import json
import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

from transaction_risk_profiler.eda.dependency_plots import partial_dependency_plots
from transaction_risk_profiler.io.dataset_utils import fill_nans
from transaction_risk_profiler.modeling.benchmark import benchmark_model

logger = logging.getLogger(__name__)


def make_partial_dependency_plots(
    model: object, cols: list[str], data_frame: pd.DataFrame, folder: str
) -> None:
    """
    Generate partial dependency plots for given columns.

    Parameters
    ----------
    model : object
        Fitted machine learning model.
    cols : list[str]
        list of columns for which to create partial dependency plots.
    data_frame : pd.DataFrame
        Data used for generating plots.
    folder : str
        Directory where to save the plots.
    """
    print("Generating partial dependency plots...")
    [partial_dependency_plots(model, col, data_frame, folder) for col in cols]
    print("Partial dependency plots generated.")


def get_data() -> pd.DataFrame:
    """
    Fetch and preprocess data from a JSON file.

    Returns
    -------
    pd.DataFrame
        Preprocessed data.
    """
    print("Fetching and preprocessing data...")
    # ... (Your existing preprocessing code here)
    print("Data fetched and preprocessed.")
    return pd.DataFrame()  # Replace this with your actual DataFrame


def get_top_features(vectorizer: TfidfVectorizer, matrix: np.ndarray) -> np.ndarray:
    """
    Get top features based on TF-IDF scores.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Fitted TfidfVectorizer object.
    matrix : np.ndarray
        TF-IDF matrix.

    Returns
    -------
    np.ndarray
        Sorted array of top features.
    """
    print("Getting top features...")
    # ... (Your existing code for getting top features)
    print("Top features obtained.")
    return np.array([])  # Replace this with your actual top features array


def get_fraud_features(top_fraud: list[str], top_desc: list[str], limit: int) -> list[str]:
    """
    Get top fraud features that are not in the top description features.

    Parameters
    ----------
    top_fraud : list[str]
        list of top fraud features.
    top_desc : list[str]
        list of top description features.
    limit : int
        Number of top features to consider.

    Returns
    -------
    list[str]
        list of top fraud features that are not in the top description features.
    """
    print("Filtering top fraud features...")
    # ... (Your existing code for filtering top fraud features)
    print("Filtered top fraud features obtained.")
    return []  # Replace this with your actual filtered top fraud features


def create_vocab(df: pd.DataFrame) -> None:
    """
    Create a vocabulary of top fraud words and save it to a JSON file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing text data.
    """
    print("Creating vocabulary...")
    # ... (Your existing code for creating vocabulary)
    print("Vocabulary created and saved.")


def read_fraud_vocab(f_name: str) -> list[str]:
    """
    Read the fraud vocabulary from a given file.

    Parameters
    ----------
    f_name : str
        Name of the file containing the fraud vocabulary.

    Returns
    -------
    list[str]
        list of fraud words.
    """
    print("Reading fraud vocabulary...")
    with open(f_name) as f:
        fraud = json.load(f)
    print("Fraud vocabulary read.")
    return fraud["fraud"]


def get_probs(
    clf: BernoulliNB, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
) -> pd.Series:
    """
    Train a Bernoulli Naive Bayes classifier, save the model, and get the
    probabilities of the positive class.

    Parameters
    ----------
    clf : BernoulliNB
        The Bernoulli Naive Bayes classifier.
    X : Union[pd.DataFrame, np.ndarray]
        Feature matrix.
    y : Union[pd.Series, np.ndarray]
        Target vector.

    Returns
    -------
    pd.Series
        Probabilities of the positive class.
    """
    print("Training Bernoulli Naive Bayes classifier...")
    clf.fit(X, y)
    with open("BernoulliNB.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Model trained and saved as 'BernoulliNB.pkl'.")

    probs = clf.predict_proba(X)[:, 1]
    print("Probabilities computed.")
    return pd.Series(probs)


# # make some plots
# This all works.  Add precision.
def plotter(results):
    indices = np.arange(len(results))
    logger.info(results)
    results = [[x[i] for x in results] for i in range(6)]
    clf_names, score, recall, precision, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Model Comparison")
    plt.barh(indices + 0.1, precision, 0.1, label="precision", color="r")
    # plt.barh(indices + .1, score, .1, label="accuracy", color='r')
    plt.barh(indices, recall, 0.1, label="recall", color="g")
    plt.barh(indices + 0.6, training_time, 0.1, label="training time", color="y")
    plt.barh(indices + 0.6, test_time, 0.1, label="test time", color="b")
    plt.yticks(())
    plt.legend(loc="best")
    plt.subplots_adjust(left=0.25)
    plt.subplots_adjust(top=0.95)
    plt.subplots_adjust(bottom=0.05)

    for i, c in zip(indices, clf_names):
        plt.text(-0.3, i, c)

    plt.show()


# for penalty in ["l2", "l1"]:
#     print('=' * 80)
#     print("%text penalty" % penalty.upper())
#     # Train Liblinear model
#     results.append(benchmark_model(LinearSVC(loss='l2', penalty=penalty,
#                                             dual=False, tol=1e-3)))
#
#     # Train SGD model
#     results.append(benchmark_model(SGDClassifier(alpha=.0001, n_iter=50,
#                                            penalty=penalty)))
#
# print('=' * 80)
# print("LinearSVC with L1-based feature selection")
# # The smaller C, the stronger the regularization.
# # The more regularization, the more sparsity.
# results.append(benchmark_model(Pipeline([
#   ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
#   ('classification', LinearSVC())
# ])))

if __name__ == "__main__":
    # df = get_data()
    # df.to_pickle("data/df.pickle")
    df = pd.read_pickle("data/df.pickle")

    # df = pd.read_pickle('trim_df.pickle')
    create_vocab(df)
    fraud_vocab = read_fraud_vocab("fraud_vocab.json")
    vectorizer = TfidfVectorizer(stop_words="english")  # , vocabulary=fraud_vocab)
    matrix = vectorizer.fit_transform(df["clean_description"])
    Xm = matrix.todense()
    ym = df["acct_type"].str.contains("fraud")

    with open("BernoulliNB.pkl") as f_un:
        Bayes = pickle.load(f_un)

    bayes_mod = get_probs(Bayes, Xm, ym)
    df["bayes_mod"] = bayes_mod

    included_cols = [
        "delivery_method",
        "event_published",
        "gts",
        "has_header",
        "org_facebook",
        "org_twitter",
        "sale_duration",
        "min_cost",
        "max_cost",
        "med_cost",
        "options",
        "ttl_sold",
        "event_country_count",
        "event_amt_count",
        "event_amt_std",
        "event_amt_min",
        "event_amt_sum",
        "event_amt_max",
        "event_amt_mean",
        "event_state_count",
        "user_country_count",
        "user_amt_min",
        "user_amt_sum",
        "user_amt_std",
        "user_amt_count",
        "user_amt_max",
        "user_amt_mean",
        "user_state_count",
        "object_id",
        "approx_payout_date",
        "body_length",
        "channels",
        "event_created",
        "event_end",
        "event_start",
        "fb_published",
        "has_analytics",
        "has_logo",
        "name_length",
        "num_order",
        "num_payouts",
        "sale_duration2",
        "show_map",
        "user_age",
        "user_created",
        "user_type",
        "desc_link_count",
        "org_link_count",
        "bayes_mod",
    ]
    fill_nans(df)
    # y = df['fraud'].values
    y = df["acct_type"].str.contains("fraud")
    X = df[included_cols].values

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    results = []
    for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(max_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=5000), "Random forest"),
        (
            SGDClassifier(alpha=0.0001, max_iter=50, penalty="elasticnet"),
            "SGDClassifier w/ elasticnet",
        ),
        (NearestCentroid(), "NearestCentroid (aka Rocchio classifier)"),
        (MultinomialNB(alpha=0.01), "Naive Bayes - Multinomial"),
        (BernoulliNB(alpha=0.01), "Naive Bayes - Bernoulli"),
        (
            GradientBoostingClassifier(
                init=None,
                learning_rate=0.1,
                loss="deviance",
                max_depth=3,
                max_features=None,
                max_leaf_nodes=None,
                min_samples_leaf=1,
                min_samples_split=2,
                min_weight_fraction_leaf=0.0,
                n_estimators=5000,
                random_state=None,
                subsample=1.0,
                verbose=0,
                warm_start=False,
            ),
            "GradientBoostingClassifier",
        ),
        (AdaBoostClassifier(n_estimators=5000), "AdaBoostClassifier"),
    ):
        logger.info("=" * 80)
        logger.info(name)
        results.append(benchmark_model(clf))
    plotter(results)
