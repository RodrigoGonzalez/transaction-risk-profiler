""" This module generates the baseline models for the transaction risk profiler. """
import json
import logging
import pickle
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
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
from sklearn.utils.extmath import density

from transaction_risk_profiler.eda.dependency_plots import partial_dependency_plots
from transaction_risk_profiler.preprocessing.text import clean_description
from transaction_risk_profiler.preprocessing.text import feature_descriptions

logger = logging.getLogger(__name__)


def fill_nans(data_frame):
    for col in data_frame:
        if data_frame[col].hasnans:
            if data_frame[col].dtype == "int64":
                data_frame[col].fillna(-999, inplace=True)
            elif data_frame[col].dtype == "float64":
                data_frame[col].fillna(-999.0, inplace=True)
            elif data_frame[col].dtype == "O":
                data_frame[col].fillna("None", inplace=True)


def make_partial_dependency_plots(model, cols, data_frame, folder):
    for col in cols:
        partial_dependency_plots(model, col, data_frame, folder)


def get_data():
    cols = [
        "approx_payout_date",
        "event_created",
        "event_end",
        "event_published",
        "event_start",
        "user_created",
    ]
    df = pd.read_json("data/transactions.json", convert_dates=cols, date_unit="s")
    del df["previous_payouts"]
    del df["ticket_types"]

    logger.info("data read in!")

    df["clean_description"] = df.apply(lambda x: clean_description(x["description"]), axis=1)
    df["clean_org"] = df.apply(lambda x: clean_description(x["org_desc"]), axis=1)

    df["clean_description"] = (
        df["clean_description"].str.lower().str.replace("[^a-z]", " ").str.replace("\n", " ")
    )
    df["clean_org"] = df["clean_org"].str.lower().str.replace("[^a-z]", " ").str.replace("\n", " ")
    logger.info("description cleaned!")

    df["desc_link_count"] = df.apply(lambda x: feature_descriptions(x["description"]), axis=1)
    df["org_link_count"] = df.apply(lambda x: feature_descriptions(x["org_desc"]), axis=1)
    logger.info("added features from description!")

    df["fraud"] = [1 if "fraud" in t else 0 for t in df["acct_type"]]
    df["payout_type_n"] = [0 if t == "" else 1 if t == "CHECK" else 2 for t in df["payout_type"]]
    currency_dict = {"USD": 0, "EUR": 1, "CAD": 2, "GBP": 3, "AUD": 4, "NZD": 5, "MXN": 6}
    df["currency_n"] = df["currency"].replace(currency_dict)
    country_dict = {None: ""}
    df["country"].replace(country_dict, inplace=True)

    return df


def get_top_features(vectorizer: TfidfVectorizer, matrix: np.ndarray) -> np.ndarray:
    features = np.array(vectorizer.get_feature_names_out())
    word_sum = np.sum(np.array(matrix.todense()), axis=0)
    return features[np.argsort(word_sum)[::-1]]


def get_fraud_features(top_fraud, top_desc, limit):
    return [x for x in top_fraud[:limit] if x not in top_desc[:limit]]


def create_vocab(df):
    fraud_vectorizer = TfidfVectorizer(stop_words="english")
    fraud_accounts = ["fraudster_event", "fraudster", "fraudster_att"]
    fraud_matrix = fraud_vectorizer.fit_transform(
        df["clean_description"][df["acct_type"].isin(fraud_accounts)]
    )

    top_fraud = get_top_features(fraud_vectorizer, fraud_matrix)

    fraud_dict = {"fraud": top_fraud[:1000].tolist()}
    f_name = "fraud_vocab.json"
    with open(f_name, "w") as f:
        json.dump(fraud_dict, f)
    logger.info("done!")
    logger.info(f"checkout {f_name}!")


def read_fraud_vocab(f_name):
    with open(f_name) as f:
        for line in f:
            fraud = json.loads(line)
    return fraud["fraud"]


def benchmark(clf):
    logger.info("_" * 80)
    logger.info("Training: ")
    logger.info(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    logger.info("train time: %0.3fs" % train_time)

    t0 = time()
    y_pred = clf.predict(X_test)
    test_time = time() - t0
    logger.info("test time: %0.3fs" % test_time)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    logger.info("accuracy:   %0.3f" % accuracy)
    logger.info("recall: %0.3f" % recall)
    logger.info("precision: %0.3f" % precision)

    if hasattr(clf, "coef_"):
        logger.info("dimensionality: %d" % clf.coef_.shape[1])
        logger.info("density: %f" % density(clf.coef_))

    logger.info()
    clf_descr = str(clf).split("(")[0]
    return clf_descr, accuracy, recall, precision, train_time, test_time


def get_probs(clf, X, y):
    clf = BernoulliNB(alpha=0.01)
    clf.fit(X, y)
    with open("BernoulliNB.pkl", "w") as f:
        pickle.dump(clf, f)
    clf.predict(X)
    probs = clf.predict_proba(X)
    return pd.Series(probs[:, 1])


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
#     results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
#                                             dual=False, tol=1e-3)))
#
#     # Train SGD model
#     results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                            penalty=penalty)))
#
# print('=' * 80)
# print("LinearSVC with L1-based feature selection")
# # The smaller C, the stronger the regularization.
# # The more regularization, the more sparsity.
# results.append(benchmark(Pipeline([
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
        results.append(benchmark(clf))
    plotter(results)
