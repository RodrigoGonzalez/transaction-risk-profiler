import itertools
import json
import logging

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from transaction_risk_profiler.modeling.baseline import try_text

logger = logging.getLogger(__name__)


STEMMER = PorterStemmer()


def clean_descr(text):
    text = text.replace("\xa0", " ")
    soup = BeautifulSoup(text, "html.parser")
    text = " ".join([try_text(x) for x in soup.contents])
    stop_words = stopwords.words("english")
    text = [x for x in text.split() if x not in stop_words]
    return " ".join([STEMMER.stem(x) for x in text])


def feature_descr(text):
    text = text.replace("\xa0", " ")
    soup = BeautifulSoup(text, "html.parser")

    links = [a.attrs["href"] for a in soup.find_all("a") if a.has_attr("href")]
    return len(links)


def get_top_features(vectorizer, matrix):
    features = np.array(vectorizer.get_feature_names())
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
    logger.info("checkout fraud_vocab.json!")


def read_fraud_vocab(f_name):
    with open(f_name) as f:
        for line in f:
            fraud = json.loads(line)
    return fraud["fraud"]


def get_pay_tic(df):
    logger.info("pay tic")
    payouts = []
    tickets = []
    for idx in range(df.shape[0]):
        payouts.extend(
            (
                df.object_id[idx],
                df.previous_payouts[idx][e]["address"],
                df.previous_payouts[idx][e]["amount"],
                df.previous_payouts[idx][e]["country"],
                df.previous_payouts[idx][e]["created"],
                df.previous_payouts[idx][e]["event"],
                df.previous_payouts[idx][e]["name"],
                df.previous_payouts[idx][e]["state"],
                df.previous_payouts[idx][e]["uid"],
                df.previous_payouts[idx][e]["zip_code"],
            )
            for e in range(len(df.previous_payouts[idx]))
        )
        tickets.extend(
            (
                df.object_id[idx],
                df.ticket_types[idx][e]["availability"],
                df.ticket_types[idx][e]["cost"],
                df.ticket_types[idx][e]["event_id"],
                df.ticket_types[idx][e]["quantity_sold"],
                df.ticket_types[idx][e]["quantity_total"],
            )
            for e in range(len(df.ticket_types[idx]))
        )
    payouts = pd.DataFrame(
        payouts,
        columns="object_id address amount country created event name state uid zip_code".split(),
    )
    tickets = pd.DataFrame(
        tickets, columns="object_id availability cost event_id quantity_sold quantity_total".split()
    )
    logger.info("created payout and ticket columns!")

    return payouts, tickets


def make_agg(item):
    """
    make specific aggregation(text) for patments_df.groupby
    """
    return {
        "amount": {
            item + "_amt_sum": "sum",
            item + "_amt_min": "min",
            item + "_amt_max": "max",
            item + "_amt_mean": "mean",
            item + "_amt_count": "count",
            item + "_amt_std": "std",
        },
        "country": {item + "_country_count": lambda x: x.nunique()},
        "state": {item + "_state_count": lambda x: x.nunique()},
    }


def feature_me(df):
    """
    input df
    output df
    """
    pay, tic = get_pay_tic(df[["object_id", "previous_payouts", "ticket_types"]])
    df.set_index("object_id", inplace=True)
    if pay.shape[0] > 0:
        event_agg = pay.groupby(["object_id"]).agg(make_agg("event")).fillna(0)

        df = df.join(event_agg["country"]).join(event_agg["amount"]).join(event_agg["state"])
    else:
        cols = [
            "_amt_sum",
            "_amt_min",
            "_amt_max",
            "_amt_mean",
            "_amt_count",
            "_amt_std",
            "_country_count",
            "_state_count",
        ]
        items = ["user", "event"]
        for item, col in itertools.product(items, cols):
            df[item + col] = 0
    if tic.shape[0] > 0:
        aggregation = {
            "quantity_sold": {"ttl_sold": "sum"},
            "cost": {
                "min_cost": "min",
                "med_cost": "median",
                "max_cost": "max",
                "options": "count",
            },
        }

        tic_agg = tic.groupby(["object_id"]).agg(aggregation)
        df = df.join(tic_agg["cost"]).join(tic_agg["quantity_sold"])
    else:
        cols = ["ttl_sold", "min_cost", "max_cost", "options", "med_cost"]
        for col in cols:
            df[col] = 0

    logger.info("aggregation feature engineering complete!")
    return df


def get_data(f_name: dict | str, head: bool = False) -> pd.DataFrame:
    if isinstance(type(f_name), str):
        df = pd.read_json(f_name).head(10) if head else pd.read_json(f_name)
    elif isinstance(type(f_name), dict):
        df = pd.Series(f_name).to_frame().T
        logger.info(df.columns)

    df = feature_me(df)
    df["clean_description"] = df.apply(lambda x: clean_descr(x["description"]), axis=1)
    df["clean_org"] = df.apply(lambda x: clean_descr(x["org_desc"]), axis=1)

    df["clean_description"] = (
        df["clean_description"].str.lower().str.replace("[^a-z]", " ").str.replace("\n", " ")
    )
    df["clean_org"] = df["clean_org"].str.lower().str.replace("[^a-z]", " ").str.replace("\n", " ")
    logger.info("description cleaned!")

    df["desc_link_count"] = df.apply(lambda x: feature_descr(x["description"]), axis=1)
    df["org_link_count"] = df.apply(lambda x: feature_descr(x["org_desc"]), axis=1)
    logger.info("added features from description!")

    return df


if __name__ == "__main__":
    data = get_data("data/train_new.json", True)

    # create_vocab(df)
    # fraud_vocab = read_fraud_vocab('fraud_vocab.json')
    # vectorizer = TfidfVectorizer(stop_words='english')  # , vocabulary=fraud_vocab)
    # matrix = vectorizer.fit_transform(df['clean_description'])
    #
    # X = matrix.todense()
    # y = df['acct_type'].str.contains('fraud')
