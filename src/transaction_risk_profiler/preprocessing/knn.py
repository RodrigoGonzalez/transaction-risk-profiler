import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

from transaction_risk_profiler.preprocessing.text import extract_text

logger = logging.getLogger(__name__)


def load_data(file_path):
    """
    Load JSON data into a DataFrame.

    Parameters
    ----------
    file_path : str
        The path to the JSON file.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_json(file_path)


def preprocess_data(df):
    """
    Preprocess the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to preprocess.

    Returns
    -------
    pd.DataFrame
    """
    df["description"] = df["description"].apply(
        lambda x: x.encode("utf-8") if x is not None else ""
    )
    df["description"] = df["description"].apply(extract_text)
    df["acct_type"] = df["acct_type"].apply(lambda x: 0 if x == "premium" else 1)
    return df[df["description"] != ""]


def vectorize_descriptions(df, column):
    """
    Vectorize the descriptions using TF-IDF.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the descriptions.
    column : str
        Column name containing the text to vectorize.

    Returns
    -------
    tuple
        Fitted vectorizer, Transformed Matrix
    """
    vectorizer = TfidfVectorizer(stop_words="english", min_df=3, norm="l2")
    X = vectorizer.fit_transform(df[column])
    return vectorizer, X


def fit_knn(X, labels, n_neighbors=10):
    """
    Fit a k-Nearest Neighbors model.

    Parameters
    ----------
    X : array-like or pd.DataFrame
        The feature matrix.
    labels : array-like
        The labels.
    n_neighbors : int, optional
        The number of neighbors to use, by default 10.

    Returns
    -------
    KNeighborsClassifier
        Fitted KNN model.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, labels)
    return knn


def main():
    df = load_data("data/transactions.json")
    tfv = df[["object_id", "acct_type", "description"]]
    del df

    tfv_no_blanks = preprocess_data(tfv)
    labels = tfv_no_blanks["description"]
    logger.info("Done processing data.")

    vectorizer, X = vectorize_descriptions(tfv_no_blanks, "description")
    _, X_all = vectorize_descriptions(tfv, "description")
    logger.info("Done vectorizing data.")

    knn = fit_knn(X, labels)
    logger.info("Done with KNN.")

    neighbors = knn.kneighbors(X_all)[1]
    logger.info("Done with knn.kneighbors.")

    tfv["description_knn"] = 0.0
    for counter, i in enumerate(range(len(tfv)), start=1):
        tfv["description_knn"][i] = tfv["acct_type"][neighbors[i]].mean()
        logger.info(counter)


if __name__ == "__main__":
    main()
