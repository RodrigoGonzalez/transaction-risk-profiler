import json

import bs4
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from sklearn.feature_extraction.text import TfidfVectorizer


def get_df(filename):
    js = json.load(open(filename))
    return json_normalize(js)


def extract_text(a):
    return bs4.BeautifulSoup(a).text


def make_tfidf(in_df, text_cols):
    df = in_df.copy()
    df["text"] = ""
    for col in text_cols:
        df["text"] += df[col]
    all_doc_text = df["text"].apply(extract_text)
    clf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True).fit(all_doc_text)
    return clf, clf.transform(all_doc_text)


def dummytize(in_df, cols):
    df = in_df.copy()
    for col in cols:
        dum_dum = pd.get_dummies(df[col])
        for c in dum_dum.columns:
            df[c] = dum_dum[c]
        df = df.drop(col, axis=1)
    return df


#
# def convert_if_present(in_df, col_vals):
#     df = in_df.copy()
#
#     for col, vals in col_vals.iteritems():
#         if col in df.columns:
#             if type(df[col][0]) is not unicode and type(df[col][0]) is not None:
#                 for val in vals:
#                     df[col] = np.where(df[col] is val, 0, df[col])
#                 df[col] = np.where(np.isnan(df[col]), 0, df[col])
#                 df[col] = np.where(df[col] == 0, 0, 1)
#             else:
#                 tmp = [1] * df.shape[0]
#                 for val in vals:
#                     tmp = np.where(df[col] == val, 0, tmp)
#                 del df[col]
#                 df[col] = tmp
#         else:
#             pass
#
#     return df


def convert_on_threshold(in_df, col_thresh):
    df = in_df.copy()
    for col, thresh in col_thresh.iteritems():
        vals = df[col].value_counts()[df[col].value_counts() <= thresh].index
        for val in vals:
            df[col] = np.where(df[col] == val, 0, df[col])
        df[col] = np.where(df[col] == 0, 0, 1)

    return df


def fix_missing_values(in_df, cols):
    df = in_df.copy()
    for col in cols:
        df[col] = df[col].fillna(0)
    return df


def drop_straight_up(in_df, cols):
    df = in_df.copy()
    for col in cols:
        if col in df.columns:
            df = df.drop(col, axis=1)
    return df


def get_rid_of_tos_lock(in_df):
    df = in_df.copy()
    df = df[df["acct_type"] != "tos_warn"]
    df = df[df["acct_type"] != "tos_lock"]
    df = df[df["acct_type"] != "locked"]
    return df


def make_label(in_df):
    df = in_df.copy()
    labels = df["acct_type"]
    labels = np.where(df["acct_type"] == "premium", 0, 1)
    del df["acct_type"]
    return df, labels
