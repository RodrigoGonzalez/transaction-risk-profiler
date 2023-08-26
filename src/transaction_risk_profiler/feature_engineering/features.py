import numpy as np
import pandas as pd


def extract_previous_payouts(x):
    if not x:
        return 0
    amount = sum(dic["amount"] or 0 for dic in x)
    return float(amount) / len(x)


cols = [
    "approx_payout_date",
    "event_created",
    "event_end",
    "event_published",
    "event_start",
    "user_created",
]
df = pd.read_json("data/transactions.json", convert_dates=cols, date_unit="s")

df.remove_columns(["object_id", "venue_country"])
df["acct_type"] = df["acct_type"].apply(lambda x: "fraud" if x != "premium" else "premium")
# df = df.unpack('ticket_types', column_name_prefix='tt')
df["num_previous_payouts"] = df["previous_payouts"].apply(lambda x: len(x))

# unpack dict columns
# df = df.to_dataframe()
# date_cols = [
#     "user_created",
#     "event_created",
#     "event_end",
#     "event_published",
#     "event_start",
#     "approx_payout_date",
# ]
# for col in date_cols:
#     df[col] = gl.SArray(pd.to_datetime(df[col], unit="s"))


df["create_lag"] = df["event_created"] - df["user_created"]
df["start_lag"] = df["event_start"] - df["event_created"]
df["payout_lag"] = df["approx_payout_date"] - df["event_created"]
df["start_payout_lag"] = df["event_start"] - df["approx_payout_date"]
# df['create_lag'] = df['event_created']-df['user_created']
# df['create_lag'] = (df['create_lag'] / np.timedelta64(1, 'D')).astype(int)
# df[date_cols] = pd.to_datetime(df[date_cols], unit='s')
# user_type to str, has_analytics, has_logo
df["venue_name"] = df["venue_name"].apply(lambda x: "" if x == "null" else x)
df["venue_name_length"] = df["venue_name"].apply(lambda x: len(x))
df["channels"] = df["channels"].astype(str)
df["org_name_length"] = df["org_name"].apply(lambda x: len(x))
df["description_length"] = df["description"].apply(lambda x: len(x))
df["num_previous_payouts"] = df["previous_payouts"].apply(lambda x: len(x))
df["avg_previous_payouts"] = df["previous_payouts"].apply(lambda x: extract_previous_payouts(x))
extractions = [
    ("max_cost", (max, "cost")),
    ("min_cost", (min, "cost")),
    ("mean_cost", (np.mean, "cost")),
    ("std_cost", (np.std, "cost")),
    ("tickets_sold", (sum, "quantity_sold")),
    ("total_tickets", (sum, "quantity_total")),
]


def get_feature(func, feat, x):
    return func([i[feat] for i in x])


for extract in extractions:
    col_name = extract[0]
    df[col_name] = df["ticket_types"].apply(lambda x: get_feature(extract[1][0], extract[1][1], x))

df["has_org_name"] = df["org_name"].apply(lambda x: "1" if x else "0")
df["has_org_desc"] = df["org_desc"].apply(lambda x: "1" if x else "0")
df["has_payee"] = df["payee_name"].apply(lambda x: "1" if x else "0")

col_cat = ["has_analytics", "has_logo", "show_map", "user_type"]
for col in col_cat:
    df[col] = df[col].astype(str)

# df["sale_duration"] = df["sale_duration"].apply(
#     lambda dict_list: int(dict_list) if dict_list else None
# )
# df["duration"] = df["sale_duration2"] - df["sale_duration"]

df.to_csv("fraud_clean.csv")

df = df.remove_columns(
    [
        "ticket_types",
        "previous_payouts",
        "description",
        "event_start",
        "event_published",
        "event_end",
        "event_created",
        "approx_payout_date",
        "user_created",
        "org_name",
        "org_desc",
        "payee_name",
        "sale_duration",
    ]
)
df.save("final.df")
