import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from transaction_risk_profiler.common.enums.dataset import TargetEnum
from transaction_risk_profiler.feature_engineering.categorical import create_binary_column
from transaction_risk_profiler.feature_engineering.categorical import create_binary_from_value
from transaction_risk_profiler.feature_engineering.simple_transforms import fill_na_with_value
from transaction_risk_profiler.feature_engineering.simple_transforms import mismatch_country
from transaction_risk_profiler.feature_engineering.simple_transforms import proportion_non_empty


def load_clean_data(filename, training=False):
    df = pd.read_json(filename)
    # df = pd.read_json('data/subset_1000.json')
    fraud_list = TargetEnum.fraud_list()

    # EDA and some preliminary feature engineering

    # creating binary categories for columns with blanks
    create_binary_from_value(df, "has_venue_address", "venue_address", "")
    create_binary_from_value(df, "has_venue_name", "venue_name", "")
    create_binary_from_value(df, "has_payee_name", "payee_name", "")
    create_binary_from_value(df, "gts_bin", "gts", 0)
    create_binary_from_value(df, "body_length_bin", "body_length", 0)
    create_binary_from_value(df, "num_payouts_bin", "num_payouts", 0)
    create_binary_column(df, "name_length_bin", "name_length", lambda x: x < 18)
    create_binary_column(df, "sale_duration_bin", "sale_duration2", lambda x: x < 10 or x == 40)

    # df['bias'] = 1

    # filling nan values
    fill_na_with_value(df, "has_header", 0)
    fill_na_with_value(df, "venue_longitude", 180)
    fill_na_with_value(df, "venue_latitude", 70)

    # making dummies for a major countries by hand
    mismatch_country(df, "mismatch_country", "country", "venue_country")

    df["is_us"] = df.country == "US"
    df["is_gb"] = df.country == "GB"
    df["is_ca"] = df.country == "CA"
    df["is_au"] = df.country == "AU"
    df["is_nz"] = df.country == "NZ"
    df["is_blank"] = df.country == ""

    # tops = df.email_domain.value_counts().head(20)
    # mask = np.logical_not(df.email_domain.isin(tops.index))
    # df.email_domain.mask(mask, other='other.com', inplace=True)

    # creating dummy columns for top 20 domains by hand
    email_domains = [
        "aol.com",
        "claytonislandtours.com",
        "comcast.net",
        "generalassemb.ly",
        "gmail.com",
        "greatworldadventures.com",
        "hotmail.co.uk",
        "hotmail.com",
        "improvboston.com",
        "kineticevents.com",
        "lidf.co.uk",
        "live.com",
        "live.fr",
        "me.com",
        "racetonowhere.com",
        "sippingnpainting.com",
        "yahoo.ca",
        "yahoo.co.uk",
        "yahoo.com",
        "ymail.com",
    ]
    for domain in email_domains:
        df[domain] = df.email_domain == domain

    # make dummies for delivery_method, channel, and email_domain
    # df = pd.concat(
    #     [
    #         df,
    #         pd.get_dummies(df.delivery_method, prefix="delivery_method"),
    #         pd.get_dummies(df["channels"], prefix="channel"),
    #         pd.get_dummies(df.currency),
    #         pd.get_dummies(df.email_domain),
    #         pd.get_dummies(df.user_type, prefix="user_type"),
    #     ],
    #     axis=1,
    # )

    # dummies by hand for delivery method, channel, currency, user type

    delivery_methods = ["delivery_method_0.0", "delivery_method_1.0"]
    for method in delivery_methods:
        df[method] = df.delivery_method == float(method.strip("delivery_method_"))

    channels = [0, 4, 5, 6, 8, 9, 10, 11, 12, 13]
    for channel in channels:
        df[f"channel_{channel}"] = df.channels == channel

    currencies = ["AUD", "CAD", "EUR", "GBP", "NZD", "USD"]
    for cur in currencies:
        df[cur] = df.currency == cur

    user_types = ["user_type_1", "user_type_2", "user_type_3", "user_type_4", "user_type_5"]
    for ut in user_types:
        df[ut] = df.user_type == int(ut.strip("user_type_"))

    df["prop_has_address"] = df["previous_payouts"].apply(
        lambda x: proportion_non_empty(x, field_name="address")
    )

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
        "delivery_method",
        "previous_payouts",
        "ticket_types",
        "payout_type",
        "channels",
        "gts",
        "body_length",
        "payee_name",
        "description",
        "listed",
        "name",
        "org_desc",
        "org_name",
        "email_domain",
        "currency",
        "user_type",
    ]
    df.drop(columns_to_drop, axis=1, inplace=True)

    if not training:
        return df
    # make fraud column
    df["fraud"] = df.acct_type.isin(fraud_list)
    df.drop("acct_type", axis=1, inplace=True)
    # create target column
    y = df.pop("fraud")

    return df, y


def make_model(X, y, thresh=0.14):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = RandomForestClassifier(n_estimators=10000, n_jobs=-1, min_samples_leaf=10)
    clf.fit(X_train, y_train)
    # preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = probs >= thresh
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    print("Recall = ", recall)
    print("Precision = ", precision)
    return clf


if __name__ == "__main__":
    X, y = load_clean_data("data/transactions.json", training=True)
    model = make_model(X.values, y.values, thresh=0.1)
    with open("model_10.pkl", "w") as f:
        pickle.dump(model, f)
