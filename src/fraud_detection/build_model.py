import pickle

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


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
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        self.model = RandomForestClassifier()

    @staticmethod
    def get_description(html):
        return BeautifulSoup(html).text

    def get_descriptions(self, df):
        return df["description"].apply(self.get_description)

    def fit(self, df):
        print("vectorizing...")
        X = self.vectorizer.fit_transform(self.get_descriptions(df)).toarray()
        print("building model...")
        self.model.fit(X, df["target"].values)
        return self

    def predict(self, df):
        X = self.vectorizer.transform(self.get_descriptions(df)).toarray()
        return self.model.predict(X)

    def predict_one(self, series):
        descriptions = [self.get_description(series["description"])]
        X = self.vectorizer.transform(descriptions).toarray()
        return self.model.predict(X)[0]


def build_model(data_filename, model_filename):
    df = get_data(data_filename)
    model = Model().fit(df)
    if model_filename:
        with open(model_filename, "w") as f:
            pickle.dump(model, f)
    return model


if __name__ == "__main__":
    build_model("data/train.json", "model.pkl")
