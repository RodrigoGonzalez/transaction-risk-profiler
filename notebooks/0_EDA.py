# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Data Prep and Preprocessing

# %load_ext autoreload
# %autoreload 2

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# +
import pandas as pd
import seaborn as sns

from transaction_risk_profiler.configs import settings

matplotlib.style.use("ggplot")
# %matplotlib inline
pd.set_option("display.max_columns", None)

# Suppress FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)
# -

# ## Load and Extract Data

df = pd.read_json(
    f"{settings.PROJECT_DIRECTORY}/{settings.DATASET_DIRECTORY}/transactions_train.json"
)

df.head()

df.info()

df.describe().T

# ###  Initially looks like there are some datetime features, so convert those to datetime

# +
datetime_features = [
    "approx_payout_date",
    "event_created",
    "event_end",
    "event_published",
    "event_start",
    "user_created",
]

df = pd.read_json(
    f"{settings.PROJECT_DIRECTORY}/{settings.DATASET_DIRECTORY}/transactions_train.json",
    convert_dates=datetime_features,
    date_unit="s",
)

# -

df.head()

df.describe().T

df.info()

# ## Initial Impressions
#
# - Looks like there are a number of text features
# - ticket_types and previous_payouts contain lists, these can be converted to
#   sub-arrays
# - venue latitude and venue longitude are geospatial features
# - listed is likely a binary feature with 'y' and 'n' corresponding to yes and
#   no, we can encode these as binary
# - fb_published, has_analytics, has_header, has_logo, show_map have min of 0
#   and max of 1, we can encode these as binary variables
# - Some of the text features can be encoded as categorical
# - org_facebook and org_twitter are likely the length of their @usernames, we
#   can bin these and see if there is a recognizable pattern
# - gts stands for gross ticket sales
# - previously identified datetime features are missing values, if they missing
#   values can be encoded as binary features
# - acct_type is our target variable
# - description is most definitely a text feature, will ignore for now
# - object_id is a row id

geospatial_features = ["venue_latitude", "venue_longitude"]
# we will include further text features, but some can probably encoded as
# categorical variables so we will keep them for now
initial_text_features = [
    "description",
    "name",
    "venue_name",
    "venue_state",
    "email_domain",
    "org_desc",
    "org_name",
    "payee_name",
    "venue_address",
]
id_columns = ["object_id"]
list_variables = ["ticket_types", "previous_payouts"]
categorical_numerical_ignore = (
    geospatial_features
    + datetime_features
    + initial_text_features
    + id_columns
    + list_variables
    + ["acct_type", "target"]
)

# ### Initial Fill In NA
# Base on the results of using value counts, determine an initial appropriate
# value to fill in NAN's

missing_values = [
    "country",
    "delivery_method",
    "has_header",
    "org_facebook",
    "org_twitter",
    "sale_duration",
]
for col in missing_values:
    print(f"Column: {col}")
    print(df[col].value_counts(dropna=False))
    print("\n")

# +
# missing_values = [
#     "country", "delivery_method", "has_header", "org_facebook", "org_twitter", "sale_duration"
# ]

# for country let's start with just using empty string
df["country"].fillna("", inplace=True)

# for sale_duration, there are negative values which don't make sense
# Make a column right away for that
df["sale_duration_neg"] = df["sale_duration"] < 0.0

# delivery method has three values and NaN, for now use 4.0, will investigate
# further
df["delivery_method"].fillna(4.0, inplace=True)


MISSING_VALS_FLOAT = ["has_header", "org_facebook", "org_twitter", "sale_duration"]

for col in MISSING_VALS_FLOAT:
    df[col].fillna(0.0, inplace=True)
    # df[f"{col}_nan"]


# -

# ## Target our target value is acct_type
# Since we are interested in identifying potentially fraudulent activity, we
# will predict on premium/non-premium account types

df.acct_type.value_counts(dropna=False)

# # Initial target encoding

df["target"] = df["acct_type"].apply(lambda x: int(x != "premium"))


# # Visualizing Data Distributions
#
# ## Significance of Visualizing Data Distributions
# Initiating any data analysis or predictive modeling project necessitates a
# comprehensive understanding of how individual variables within the dataset
# are dispersed. Employing visualization techniques to depict these
# distributions serves as an invaluable tool, offering immediate insights that
# are crucial for guiding subsequent phases of analysis.
#
# ## Key Questions Answered Through Visualization
# Visual depictions of data distributions are instrumental in swiftly
# addressing a multitude of pivotal questions that are foundational to the
# analytic process:
#
# - Extent of Observational Data: What is the minimum and maximum range that
#   the dataset spans?
# - Central Tendency Measures: Where does the bulk of the data concentrate, as
#   indicated by statistical measures such as the mean, median, or mode?
# - Skewness Assessment: Is the dataset predominantly skewed towards higher or
#   lower values?
# - Evidence of Multiple Peaks: Does the data reveal bimodal or multimodal
#   tendencies, suggesting multiple underlying processes or groups?
# - Identification of Outliers: Are there any extreme values that deviate
#   significantly from the majority of observations?
#
# ## Subset-Specific Insights
# Moreover, it's imperative to examine whether these key characteristics
# persist or vary when the data is partitioned based on other influencing
# factors or variables. For instance, does the distribution of a variable like
# income differ when categorized by variables such as age, gender, or
# geographic location?

# # Categorical and Numerical Variables
#
# Explore what variables we can probably encode as categorical or binary
# variables
#
# - We will use Counter to see how many unique values each column has
# - Given that we know which columns are datetime, we can skip these and the
#   geo_spatial features
# - We can plot the features that have unique values <= 20 as bar charts to
#   further investigate
# - If > 20, let's bin, number of bins will be set the sqrt of num unique, if
#   num unique is greater than 50

# +

for i, col in enumerate(df.columns):
    if col not in categorical_numerical_ignore:  #  + ["num_payouts", "num_order"]
        print(f"Charting Column {i}: {col}")
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create the displot on the specific Axes
        if df[col].value_counts(dropna=False).shape[0] > 20 and isinstance(df[col], (int, float)):
            bins = np.linspace(min(df[col]), max(df[col]), 20)
            sns.histplot(df, x=col, hue="target", multiple="dodge", ax=ax, bins=bins)
        else:
            sns.histplot(df, x=col, hue="target", multiple="dodge", ax=ax)

        # Use the set_title method on the Axes object
        ax.set_title(f"{' '.join(col.capitalize().split('_'))} Column by Target")
        ax.set_ylabel("Frequency")
        ax.set_xlabel(col)
        if col in ["country", "venue_country"]:
            plt.xticks(rotation=90)
        plt.grid()
        plt.show()


# +
# Exploratory Data Analysis and Preliminary Feature Engineering

fraud_list = ["fraudster", "fraudster_event", "fraudster_att"]
spammer_list = ["spammer_limited", "spammer_noinvite", "spammer_web", "spammer", "spammer_warn"]
tos_list = ["tos_warn", "tos_lock"]
premium_list = ["premium"]
locked_list = ["locked"]
suspicious_list = spammer_list + tos_list + locked_list
fraud_list += suspicious_list

# Classify records as fraud or not based on the 'acct_type' column.
# A new boolean column 'fraud' is created where True indicates fraud and False
# indicates not fraud.
df["fraud"] = df.acct_type.isin(fraud_list)

# Convert Unix timestamps to human-readable datetime format for various
# event-related columns.
# Separate the date and time into new columns for further analysis.

# Convert 'approx_payout_date' and create separate columns for date and time.
df.approx_payout_date = pd.to_datetime(df.approx_payout_date, unit="s")
df["approx_payout_date_date"] = df.approx_payout_date.dt.date
df["approx_payout_date_time"] = df.approx_payout_date.dt.time

# Convert 'event_created' and create separate columns for date and time.
df.event_created = pd.to_datetime(df.event_created, unit="s")
df["event_created_date"] = df.event_created.dt.date
df["event_created_time"] = df.event_created.dt.time

# Convert 'event_published' and create separate columns for date and time.
df.event_published = pd.to_datetime(df.event_published, unit="s")
df["event_publish_date"] = df.event_published.dt.date
df["event_publish_time"] = df.event_published.dt.time

# Convert 'event_start' and create separate columns for date and time.
df.event_start = pd.to_datetime(df.event_start, unit="s")
df["event_start_date"] = df.event_start.dt.date
df["event_start_time"] = df.event_start.dt.time

# Convert 'event_end' and create separate columns for date and time.
df.event_end = pd.to_datetime(df.event_end, unit="s")
df["event_end_date"] = df.event_end.dt.date
df["event_end_time"] = df.event_end.dt.time

# Calculate event duration in hours.
df["event_duration_h"] = (df.event_end - df.event_start).astype("timedelta64[s]")

# Convert 'user_created' and create separate columns for date and time.
df.user_created = pd.to_datetime(df.user_created, unit="s")
df["user_create_date"] = df.user_created.dt.date
df["user_create_time"] = df.user_created.dt.time

# Create boolean columns to indicate the presence of venue address, venue name,
# and payee name.
df["has_venue_address"] = df.venue_address != ""
df["has_venue_name"] = df.venue_name != ""
df["has_payee_name"] = df.payee_name != ""

# Fill missing values in 'has_header' column with 0.
df["has_header"].fillna(value=0, inplace=True)

df["has_payee_name"] = df.payee_name != ""
df["has_header_filled"] = df.has_header
df["has_header_filled"].fillna(value=2, inplace=True)

# "country": {f"{item}_country_count": lambda x: x.nunique()},
# "state": {f"{item}_state_count": lambda x: x.nunique()},

# +
# Add new datetime features to list of datetime features
# additional_datetime_features = []
# for feat in datetime_features:
#     datetime_features.append(f"{feat}_date")
#     datetime_features.append(f"{feat}_time")

# datetime_features.extend(additional_datetime_features)
# -

df.hist(figsize=(20, 20))
plt.show()

df.describe().T

previous_payouts = df.previous_payouts.head(3).values
for payouts in previous_payouts:
    if payouts:
        for payout in payouts:
            print("\n")
            print(payout)

# # List Variables
#
#

# Lets first add columns for the rows with empty lists
df["ticket_types_empty"] = df["ticket_types"].apply(lambda x: 1 if not x else 0)
df["previous_payouts_empty"] = df["previous_payouts"].apply(lambda x: 1 if not x else 0)


ticket_types = df.ticket_types.head(10).values
for tickets in ticket_types:
    if tickets:
        for ticket in tickets:
            print("\n")
            print(ticket)

# +
# Aggregate List Data
from transaction_risk_profiler.preprocessing.aggregations import get_payout_info
from transaction_risk_profiler.preprocessing.aggregations import get_ticket_info

# Step 1: Extract Payout and Ticket Information
payout_df = get_payout_info(df)
ticket_df = get_ticket_info(df)

payout_df.head(), ticket_df.head()

# +
from transaction_risk_profiler.preprocessing.aggregations import create_agg_schema

# Step 2: Create Aggregation Schema
payout_agg_schema = create_agg_schema("user")
ticket_agg_schema = create_agg_schema("event")

print(payout_agg_schema, ticket_agg_schema)

# +
from transaction_risk_profiler.preprocessing.aggregations import aggregate_data

# Step 3: Aggregate Data
agg_payout_df = aggregate_data(payout_df, "object_id", payout_agg_schema)
agg_ticket_df = aggregate_data(ticket_df, "object_id", ticket_agg_schema)
# -

agg_payout_df.head()

agg_ticket_df.head()

# Get new columns
agg_payout_cols = list(agg_payout_df.columns)
agg_ticket_cols = list(agg_ticket_df.columns)
all_agg_columns = agg_payout_cols + agg_ticket_cols
all_agg_columns.append("target")
agg_payout_cols.append("target")
agg_ticket_cols.append("target")

# Perform a left join between df and agg_payout_df on the 'object_id' column
from copy import deepcopy

df = deepcopy(df)
merged_df_0 = df.merge(agg_payout_df, on="object_id", how="left", copy=True, suffixes=(None, None))
merged_df = merged_df_0.merge(
    agg_ticket_df, on="object_id", how="left", copy=True, suffixes=(None, None)
)
merged_df.fillna(0)

merged_df.head()

sns.pairplot(merged_df[agg_payout_cols])
plt.show()

sns.pairplot(merged_df[agg_ticket_cols])
plt.show()

# ## Binning
#
# Let's see if there is a discernible patter for the length of the facebook and
# twitter handles

# facebook
df["org_facebook"][df.fraud is True].hist(bins=50, figsize=(12, 8))
df["org_facebook"][df.fraud is False].hist(bins=50, figsize=(12, 8))
plt.show()

# twitter
comparative_histogram(
    df[df.fraud is True]["org_twitter"],
    df[df.fraud is False]["org_twitter"],
    col_name="org_twitter",
    max_val=60,
    label_1="fraudulent transaction",
    label_2="not fraudulent transaction",
)


# Longitude and Latitude
df.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    grid=True,
    s=df["fraud"],
    label="population",
    c="median_house_value",
    cmap="jet",
    colorbar=True,
    legend=True,
    sharex=False,
    figsize=(10, 7),
)
