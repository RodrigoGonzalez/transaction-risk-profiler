""" This module contains functions for preprocessing text data with TF-IDF. """
import json

import numpy as np
import pandas as pd
from pandas import json_normalize
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from transaction_risk_profiler.preprocessing.text import extract_text


def get_df(filename):
    js = json.load(open(filename))
    return json_normalize(js)


def generate_tfidf_features(
    in_df: pd.DataFrame, text_cols: list[str]
) -> tuple[TfidfVectorizer, csr_matrix]:
    """
    Generates TF-IDF features from the text columns of the input DataFrame.

    This function first concatenates the text from all specified columns into a
    single string for each row in the DataFrame.
    It then applies text extraction to this concatenated text. A TF-IDF
    Vectorizer is fitted to the extracted text and used to transform the text
    into TF-IDF features.

    Parameters
    ----------
    in_df : pandas.DataFrame
        The input DataFrame.
    text_cols : list[str]
        A list of column names from which text is to be extracted.

    Returns
    -------
    vectorizer : TfidfVectorizer
        The fitted TF-IDF Vectorizer.
    tfidf_features : csr_matrix
        The TF-IDF features generated from the extracted text.
    """
    df = in_df.copy()
    df["text"] = df[text_cols].apply(" ".join, axis=1)
    all_doc_text = df["text"].apply(extract_text)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    vectorizer.fit(all_doc_text)
    tfidf_features = vectorizer.transform(all_doc_text)
    return vectorizer, tfidf_features


def dummytize(in_df, cols):
    df = in_df.copy()
    for col in cols:
        dum_dum = pd.get_dummies(df[col])
        for c in dum_dum.columns:
            df[c] = dum_dum[c]
        df = df.drop(col, axis=1)
    return df


def convert_if_present(in_df: pd.DataFrame, col_vals: dict) -> pd.DataFrame:
    """
    Converts specified values in the input DataFrame to 0 or 1.

    For each column specified in col_vals, if the column exists in the DataFrame
    and its first value is not a string or None, the function converts the
    specified values to 0 and all other values to 1. If the first value is a
    string or None, all specified values are converted to 0 and all other values
    are converted to 1. If the column does not exist in the DataFrame, the
    function does nothing.

    Parameters
    ----------
    in_df : pandas.DataFrame
        The input DataFrame.
    col_vals : dict
        A dictionary where keys are column names and values are lists of values
        to be converted.

    Returns
    -------
    df : pandas.DataFrame
        The modified DataFrame.
    """
    df = in_df.copy()

    for col, vals in col_vals.items():
        if col in df.columns:
            if isinstance(df[col].iloc[0], str) or df[col].iloc[0] is None:
                df[col] = df[col].apply(lambda x: 0 if x in vals else 1)
            else:
                df[col] = df[col].apply(lambda x: 0 if x in vals or pd.isnull(x) else 1)

    return df


def convert_on_threshold(in_df: pd.DataFrame, col_thresh: dict[str, int]) -> pd.DataFrame:
    """
    Converts values in specified columns of the input DataFrame based on a
    threshold.

    For each column specified in col_thresh, values that appear less than or
    equal to the threshold times are converted to 0.

    All other values are converted to 1.

    Parameters
    ----------
    in_df : pandas.DataFrame
        The input DataFrame.
    col_thresh : dict
        A dictionary where keys are column names and values are thresholds.

    Returns
    -------
    df : pandas.DataFrame
        The modified DataFrame.
    """
    df = in_df.copy()
    for col, thresh in col_thresh.items():
        vals = df[col].value_counts().loc[lambda x: x <= thresh].index
        df[col] = df[col].apply(lambda x: 0 if x in vals else 1)
    return df


def fix_missing_values(in_df, cols):
    df = in_df.copy()
    for col in cols:
        df[col] = df[col].fillna(0)
    return df


def make_label(in_df: pd.DataFrame, col: str, val: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generates a label for a given input DataFrame based on a specified column
    and value.

    Parameters
    ----------
    in_df : pandas.DataFrame
        The input DataFrame.
    col : str
        The column based on which labels are to be generated.
    val : str
        The value in the column that is to be assigned a label of 0. All other
        values are assigned a label of 1.

    Returns
    -------
    df : pandas.DataFrame
        The modified input DataFrame with the specified column removed.
    labels : numpy.ndarray
        An array of labels generated based on the specified column and value.
    """
    df = in_df.copy()
    labels = df[col]
    labels = np.where(df[col] == val, 0, 1)
    df.drop(columns=[col], inplace=True)
    return df, labels
