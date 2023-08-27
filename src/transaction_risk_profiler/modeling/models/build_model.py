import logging
import pickle

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def is_fraud(x):
    return x.startswith("fraud")


def get_data(filepath):
    df = pd.read_json(filepath)
    mask = np.logical_or(df["acct_type"] == "premium", df["acct_type"].apply(is_fraud))
    df = df[mask]
    df["target"] = df["acct_type"].apply(lambda x: x.startswith("fraud"))
    df.pop("acct_type")  # drop column we're trying to predict
    df.pop("passwd")  # useless column
    return df


class Model:
    """Model for predicting fraudulent transactions."""

    def __init__(self, n_features: int = 1000):
        """
        Initialize the Model with a TfidfVectorizer and a RandomForestClassifier.

        Parameters
        ----------
        n_features : int, optional
            The maximum number of features for TfidfVectorizer.
            Defaults to 1000.
        """
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=n_features)
        self.model = RandomForestClassifier()

    @staticmethod
    def get_description(html: str) -> str:
        """
        Extract and return text from HTML.

        Parameters
        ----------
        html : str
            The HTML content.

        Returns
        -------
        str
            The extracted text.
        """
        return BeautifulSoup(html).text

    def get_descriptions(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply text extraction to a DataFrame column.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing HTML descriptions.

        Returns
        -------
        pd.Series
            Series with extracted text.
        """
        return df["description"].apply(self.get_description)

    def fit(self, df: pd.DataFrame) -> "Model":
        """
        Fit the model to a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to fit.

        Returns
        -------
        Model
            The fitted model.
        """
        logger.info("Vectorizing...")
        X = self.vectorizer.fit_transform(self.get_descriptions(df)).toarray()

        logger.info("Building model...")
        self.model.fit(X, df["target"].values)

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict the target for a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame for which to predict targets.

        Returns
        -------
        np.ndarray
            The predicted targets.
        """
        X = self.vectorizer.transform(self.get_descriptions(df)).toarray()
        return self.model.predict(X)

    def predict_one(self, series: pd.Series) -> int:
        """
        Predict the target for a single series.

        Parameters
        ----------
        series : pd.Series
            The series for which to predict the target.

        Returns
        -------
        int
            The predicted target.
        """
        descriptions = [self.get_description(series["description"])]
        data = self.vectorizer.transform(descriptions).toarray()
        return self.model.predict(data)[0]


def build_model(data_filename, model_filename):
    df = get_data(data_filename)
    model = Model().fit(df)
    if model_filename:
        with open(model_filename, "w") as f:
            pickle.dump(model, f)
    return model


if __name__ == "__main__":
    build_model("data/train.json", "model.pkl")
