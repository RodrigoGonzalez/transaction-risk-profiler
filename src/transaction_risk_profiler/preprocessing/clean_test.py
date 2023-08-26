import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split


def prop(cells):
    if len(cells) <= 0:
        return 0
    return 1 - (sum(cell["address"].strip() == "" for cell in cells) / float(len(cells)))


def load_clean_data(filename, training=False):
    df = pd.read_json(filename)
    # df = pd.read_json('data/subset_1000.json')
    fraud_list = ["fraudster", "fraudster_event", "fraudster_att"]
    spammer_list = ["spammer_limited", "spammer_noinvite", "spammer_web", "spammer", "spammer_warn"]
    tos_list = ["tos_warn", "tos_lock"]
    locked_list = ["locked"]
    suspicious_list = spammer_list + tos_list + locked_list
    fraud_list += suspicious_list

    # EDA and some preliminary feature engineering

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

    df["prop_has_address"] = df["previous_payouts"].apply(prop)

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
    recall_score(y_test, preds)
    precision_score(y_test, preds)
    # print "Recall = ", recall
    # print "Precision = ", precision
    return clf


if __name__ == "__main__":
    X, y = load_clean_data("data/transactions.json", training=True)
    model = make_model(X.values, y.values, thresh=0.1)
    with open("model_10.pkl", "w") as f:
        pickle.dump(model, f)
