import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from transaction_risk_profiler.common.enums.dataset import TargetEnum
from transaction_risk_profiler.preprocessing.feature_engineering import proportion_non_empty

logger = logging.getLogger(__name__)


def load_clean_data():
    df = pd.read_json("data/transactions.json")
    # df = pd.read_json('data/subset_1000.json')
    fraud_list = TargetEnum.fraud_list()

    # EDA and some preliminary feature engineering
    df["fraud"] = df.acct_type.isin(fraud_list)  # classify as fraud or not (True or False)
    # creating binary categories for columns with blanks
    df["has_venue_address"] = df.venue_address != ""
    df["has_venue_name"] = df.venue_name != ""
    df["has_payee_name"] = df.payee_name != ""
    df["gts_bin"] = df.gts != 0
    df["body_length_bin"] = df.body_length != 0
    df["name_length_bin"] = df.name_length < 18
    df["num_payouts_bin"] = df.num_payouts != 0
    df["sale_duration_bin"] = df.sale_duration2.apply(lambda x: x < 10 or x == 40)

    # df['bias'] = 1

    # filling nan values
    df["has_header"].fillna(value=0, inplace=True)
    df.venue_longitude.fillna(180, inplace=True)
    df.venue_latitude.fillna(70, inplace=True)

    # making dummies for a major countries by hand
    df["mismatch_country"] = df.country != df.venue_country
    df["is_us"] = df.country == "US"
    df["is_gb"] = df.country == "GB"
    df["is_ca"] = df.country == "CA"
    df["is_au"] = df.country == "AU"
    df["is_nz"] = df.country == "NZ"
    df["is_blank"] = df.country == ""

    # # make dummies by hand for three biggest email clients
    # df['gmail'] = df.email_domain == 'gmail.com'
    # df['yahoo'] = df.email_domain == 'yahoo.com'
    # df['hotmail'] = df.email_domain == 'hotmail.com'

    tops = df.email_domain.value_counts().head(20)
    mask = np.logical_not(df.email_domain.isin(tops.index))
    df.email_domain.mask(mask, other="other.com", inplace=True)

    # make dummies for delivery_method, channel, and email_domain
    df = pd.concat(
        [
            df,
            pd.get_dummies(df.delivery_method, prefix="delivery_method"),
            pd.get_dummies(df["channels"], prefix="channel"),
            pd.get_dummies(df.currency),
            pd.get_dummies(df.email_domain),
            pd.get_dummies(df.user_type, prefix="user_type"),
        ],
        axis=1,
    )

    df["prop_has_address"] = df["previous_payouts"].apply(
        lambda x: proportion_non_empty(x, field_name="address")
    )

    # df1 = df[
    #     [
    #         "gts_bin",
    #         "name_length",
    #         "venue_latitude",
    #         "venue_longitude",
    #         "mismatch_country",
    #         "channel__0",
    #         "channel__4",
    #         "channel__5",
    #         "channel__6",
    #         "channel__7",
    #         "channel__8",
    #         "channel__9",
    #         "channel__10",
    #         "channel__11",
    #         "channel__12",
    #         "is_us",
    #         "is_gb",
    #         "is_ca",
    #         "is_au",
    #         "is_nz",
    #         "is_blank",
    #         "delivery_method_0.0",
    #         "delivery_method_1.0",
    #     ]
    # ]

    # dropping columns
    columns_to_drop = [
        "approx_payout_date",
        "event_created",
        "event_published",
        "event_start",
        "event_end",
        "user_created",
        "venue_state",
        "venue_name",
        "venue_address",
        "venue_country",
        "venue_longitude",
        "venue_latitude",
        "sale_duration",
        "sale_duration2",
        "name_length",
        "num_payouts",
        "org_twitter",
        "org_facebook",
        "country",
        "other.com",
        "delivery_method",
        "previous_payouts",
        "ticket_types",
        "payout_type",
        "channels",
        "delivery_method_3.0",
        "gts",
        "body_length",
        "payee_name",
        "acct_type",
        "description",
        "listed",
        "name",
        "org_desc",
        "org_name",
        "email_domain",
        "currency",
        "MXN",
        "user_type",
        "user_type_5",
    ]
    df.drop(columns_to_drop, axis=1, inplace=True)

    # create target column
    y = df.pop("fraud")

    return df, y


def make_dumb_model(X, y, thresh=0.14):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = RandomForestClassifier(n_estimators=10000, n_jobs=-1)
    clf.fit(X_train, y_train)
    # preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = probs >= thresh
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    logger.info("Recall = ", recall)
    logger.info("Precision = ", precision)


if __name__ == "__main__":
    X, y = load_clean_data()
    make_dumb_model(X.values, y.values, thresh=0.1)
