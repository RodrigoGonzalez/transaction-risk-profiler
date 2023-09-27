""" Training pipeline. """
import tarfile
import urllib.request
from pathlib import Path
from zlib import crc32

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy import stats
from scipy.stats import binom
from scipy.stats import expon
from scipy.stats import geom
from scipy.stats import loguniform
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.estimator_checks import check_estimator

from transaction_risk_profiler.preprocessing.feature_engineering.features import (
    FeatureFromRegressor,
)
from transaction_risk_profiler.preprocessing.scaling import StandardScalerClone

# Get the Data


def load_dataset_data():
    tarball_path = Path("datasets/dataset.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/dataset.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as dataset_tarball:
            dataset_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/dataset/dataset.csv"))


dataset = load_dataset_data()

dataset.head()
dataset.info()

dataset["ocean_proximity"].value_counts()

dataset.describe()


# ## Create a Test Set

# In[11]:


def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[12]:


train_set, test_set = shuffle_and_split_data(dataset, 0.2)
len(train_set)
len(test_set)


np.random.seed(42)


# Sadly, this won't guarantee that this notebook will output exactly the same
# results as in the book, since there are other possible sources of variation.
# The most important is the fact that algorithms get tweaked over time when
# libraries evolve. So please tolerate some minor differences: hopefully, most
# of the outputs should be the same, or at least in the right ballpark.

# Note: another source of randomness is the order of Python sets: it is based
# on Python's `hash()` function, which is randomly "salted" when Python starts
# up (this started in Python 3.3, to prevent some denial-of-service attacks).
# To remove this randomness, the solution is to set the `PYTHONHASHSEED`
# environment variable to `"0"` _before_ Python even starts up. Nothing will
# happen if you do it after that. Luckily, if you're running this notebook on
# Colab, the variable is already set for you.

# In[15]:


def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32


def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# In[16]:


dataset_with_id = dataset.reset_index()  # adds an `index` column
train_set, test_set = split_data_with_id_hash(dataset_with_id, 0.2, "index")


# In[17]:


dataset_with_id["id"] = dataset["longitude"] * 1000 + dataset["latitude"]
train_set, test_set = split_data_with_id_hash(dataset_with_id, 0.2, "id")


# In[18]:


train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)


# In[19]:


test_set["total_bedrooms"].isnull().sum()


# To find the probability that a random sample of 1,000 people contains less
# than 48.5% female or more than 53.5% female when the population's female
# ratio is 51.1%, we use the
# [binomial distribution](https://en.wikipedia.org/wiki/Binomial_distributionution).
# The `cdf()` method of the binomial distribution gives us the probability that
# the number of females will be equal or less than the given value.

# In[20]:


# extra code – shows how to compute the 10.7% proba of getting a bad sample


sample_size = 1000
ratio_female = 0.511
proba_too_small = binom(sample_size, ratio_female).cdf(485 - 1)
proba_too_large = 1 - binom(sample_size, ratio_female).cdf(535)
print(proba_too_small + proba_too_large)


# If you prefer simulations over maths, here's how you could get roughly the
# same result:

# In[21]:


# extra code – shows another way to estimate the probability of bad sample

np.random.seed(42)

samples = (np.random.rand(100_000, sample_size) < ratio_female).sum(axis=1)
((samples < 485) | (samples > 535)).mean()

dataset["income_cat"] = pd.cut(
    dataset["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5]
)

dataset["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()


splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
stratified_splits = []
for train_index, test_index in splitter.split(dataset, dataset["income_cat"]):
    stratified_train_set_n = dataset.iloc[train_index]
    stratified_test_set_n = dataset.iloc[test_index]
    stratified_splits.append([stratified_train_set_n, stratified_test_set_n])

stratified_train_set, stratified_test_set = stratified_splits[0]


# It's much shorter to get a single stratified split:


stratified_train_set, stratified_test_set = train_test_split(
    dataset, test_size=0.2, stratify=dataset["income_cat"], random_state=42
)


stratified_test_set["income_cat"].value_counts() / len(stratified_test_set)


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

compare_props = pd.DataFrame(
    {
        "Overall %": income_cat_proportions(dataset),
        "Stratified %": income_cat_proportions(stratified_test_set),
        "Random %": income_cat_proportions(test_set),
    }
).sort_index()
compare_props.index.name = "Income Category"
compare_props["Strat. Error %"] = compare_props["Stratified %"] / compare_props["Overall %"] - 1
compare_props["Rand. Error %"] = compare_props["Random %"] / compare_props["Overall %"] - 1
(compare_props * 100).round(2)


for set_ in (stratified_train_set, stratified_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# Discover and Visualize the Data to Gain Insights


dataset = stratified_train_set.copy()


# Visualizing Geographical Data

dataset.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
plt.show()


# In[33]:


dataset.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    grid=True,
    s=dataset["population"] / 100,
    label="population",
    c="median_house_value",
    cmap="jet",
    colorbar=True,
    legend=True,
    sharex=False,
    figsize=(10, 7),
)
plt.show()


# The argument `sharex=False` fixes a display bug: without it, the x-axis
# values and label are not displayed
# (see: https://github.com/pandas-dev/pandas/issues/10611).

# The next cell generates the first figure in the chapter (this code is not in
# the book). It's just a beautified version of the previous figure, with an
# image of California added in the background, nicer label names and no grid.

# In[34]:


# extra code – this cell generates the first figure in the chapter

# Download the California image

dataset_renamed = dataset.rename(
    columns={
        "latitude": "Latitude",
        "longitude": "Longitude",
        "population": "Population",
        "median_house_value": "Median house value (ᴜsᴅ)",
    }
)
dataset_renamed.plot(
    kind="scatter",
    x="Longitude",
    y="Latitude",
    s=dataset_renamed["Population"] / 100,
    label="Population",
    c="Median house value (ᴜsᴅ)",
    cmap="jet",
    colorbar=True,
    legend=True,
    sharex=False,
    figsize=(10, 7),
)


# ## Looking for Correlations

# In[35]:


corr_matrix = dataset.corr()


# In[36]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# In[37]:


attributes = ["median_house_value", "median_income", "total_rooms", "dataset_median_age"]
scatter_matrix(dataset[attributes], figsize=(12, 8))
plt.show()


# In[38]:


dataset.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)
plt.show()


# ## Experimenting with Attribute Combinations

# In[39]:


dataset["rooms_per_house"] = dataset["total_rooms"] / dataset["households"]
dataset["bedrooms_ratio"] = dataset["total_bedrooms"] / dataset["total_rooms"]
dataset["people_per_house"] = dataset["population"] / dataset["households"]


# In[40]:


corr_matrix = dataset.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# # Prepare the Data for Machine Learning Algorithms

# Let's revert to the original training set and separate the target (note that
# `stratified_train_set.drop()` creates a copy of `stratified_train_set` without the
# column, it doesn't actually modify `stratified_train_set` itself, unless you pass
# `inplace=True`):

# In[41]:


dataset = stratified_train_set.drop("median_house_value", axis=1)
dataset_labels = stratified_train_set["median_house_value"].copy()


# ## Data Cleaning

# In the book 3 options are listed to handle the NaN values:
#
# ```python
# dataset.dropna(subset=["total_bedrooms"], inplace=True)    # option 1
#
# dataset.drop("total_bedrooms", axis=1)       # option 2
#
# median = dataset["total_bedrooms"].median()  # option 3
# dataset["total_bedrooms"].fillna(median, inplace=True)
# ```
#
# For each option, we'll create a copy of `dataset` and work on that copy to
# avoid breaking `dataset`. We'll also show the output of each option, but
# filtering on the rows that originally contained a NaN value.

# In[42]:


null_rows_idx = dataset.isnull().any(axis=1)
dataset.loc[null_rows_idx].head()


dataset_option1 = dataset.copy()
dataset_option1.dropna(subset=["total_bedrooms"], inplace=True)  # option 1
dataset_option1.loc[null_rows_idx].head()


dataset_option2 = dataset.copy()
dataset_option2.drop("total_bedrooms", axis=1, inplace=True)  # option 2
dataset_option2.loc[null_rows_idx].head()


dataset_option3 = dataset.copy()

median = dataset["total_bedrooms"].median()
dataset_option3["total_bedrooms"].fillna(median, inplace=True)  # option 3
dataset_option3.loc[null_rows_idx].head()


# Separating out the numerical attributes to use the `"median"` strategy (as
# it cannot be calculated on text attributes like `ocean_proximity`):

dataset_num = dataset.select_dtypes(include=[np.number])
imputer = SimpleImputer(strategy="median")
imputer.fit(dataset_num)
imputer.statistics_


# Check that this is the same as manually computing the median of each attribute:


dataset_num.median().values


# Transform the training set:


X = imputer.transform(dataset_num)


imputer.feature_names_in_


dataset_tr = pd.DataFrame(X, columns=dataset_num.columns, index=dataset_num.index)


dataset_tr.loc[null_rows_idx].head()


imputer.strategy


dataset_tr = pd.DataFrame(X, columns=dataset_num.columns, index=dataset_num.index)


dataset_tr.loc[null_rows_idx].head()  # not shown in the book


# Now let's drop some outliers:


isolation_forest = IsolationForest(random_state=42)
outlier_pred = isolation_forest.fit_predict(X)


# If you wanted to drop outliers, you would run the following code:


dataset = dataset.iloc[outlier_pred == 1]
dataset_labels = dataset_labels.iloc[outlier_pred == 1]


# ## Handling Text and Categorical Attributes

# Now let's preprocess the categorical input feature, `ocean_proximity`:


dataset_cat = dataset[["ocean_proximity"]]
dataset_cat.head(8)


ordinal_encoder = OrdinalEncoder()
dataset_cat_encoded = ordinal_encoder.fit_transform(dataset_cat)


dataset_cat_encoded[:8]


ordinal_encoder.categories_


cat_encoder = OneHotEncoder()
dataset_cat_1hot = cat_encoder.fit_transform(dataset_cat)


dataset_cat_1hot


# By default, the `OneHotEncoder` class returns a sparse array, but we can
# convert it to a dense array if needed by calling the `toarray()` method:


dataset_cat_1hot.toarray()


# Alternatively, you can set `sparse=False` when creating the `OneHotEncoder`:


cat_encoder = OneHotEncoder(sparse=False)
dataset_cat_1hot = cat_encoder.fit_transform(dataset_cat)
dataset_cat_1hot


cat_encoder.categories_


df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
pd.get_dummies(df_test)


cat_encoder.transform(df_test)


df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
pd.get_dummies(df_test_unknown)


# In[74]:


cat_encoder.handle_unknown = "ignore"
cat_encoder.transform(df_test_unknown)


# In[75]:


cat_encoder.feature_names_in_


# In[76]:


cat_encoder.get_feature_names_out()


# In[77]:


df_output = pd.DataFrame(
    cat_encoder.transform(df_test_unknown),
    columns=cat_encoder.get_feature_names_out(),
    index=df_test_unknown.index,
)


# In[78]:


df_output


# ## Feature Scaling

# In[79]:


min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
dataset_num_min_max_scaled = min_max_scaler.fit_transform(dataset_num)


# In[80]:


std_scaler = StandardScaler()
dataset_num_std_scaled = std_scaler.fit_transform(dataset_num)


# In[81]:


# extra code – this cell generates Figure 2–17
fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
dataset["population"].hist(ax=axs[0], bins=50)
dataset["population"].apply(np.log).hist(ax=axs[1], bins=50)
axs[0].set_xlabel("Population")
axs[1].set_xlabel("Log of population")
axs[0].set_ylabel("Number of districts")
plt.show()


# What if we replace each value with its percentile?

# In[82]:


# extra code – just shows that we get a uniform distribution
percentiles = [np.percentile(dataset["median_income"], p) for p in range(1, 100)]
flattened_median_income = pd.cut(
    dataset["median_income"], bins=[-np.inf] + percentiles + [np.inf], labels=range(1, 100 + 1)
)
flattened_median_income.hist(bins=50)
plt.xlabel("Median income percentile")
plt.ylabel("Number of districts")
plt.show()
# Note: incomes below the 1st percentile are labeled 1, and incomes above the
# 99th percentile are labeled 100. This is why the distribution below ranges
# from 1 to 100 (not 0 to 100).


# In[83]:


age_simil_35 = rbf_kernel(dataset[["dataset_median_age"]], [[35]], gamma=0.1)


# In[84]:


# extra code – this cell generates Figure 2–18

ages = np.linspace(
    dataset["dataset_median_age"].min(), dataset["dataset_median_age"].max(), 500
).reshape(-1, 1)
gamma1 = 0.1
gamma2 = 0.03
rbf1 = rbf_kernel(ages, [[35]], gamma=gamma1)
rbf2 = rbf_kernel(ages, [[35]], gamma=gamma2)

fig, ax1 = plt.subplots()

ax1.set_xlabel("Housing median age")
ax1.set_ylabel("Number of districts")
ax1.hist(dataset["dataset_median_age"], bins=50)

ax2 = ax1.twinx()  # create a twin axis that shares the same x-axis
color = "blue"
ax2.plot(ages, rbf1, color=color, label="gamma = 0.10")
ax2.plot(ages, rbf2, color=color, label="gamma = 0.03", linestyle="--")
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_ylabel("Age similarity", color=color)

plt.legend(loc="upper left")
plt.show()


# In[85]:


target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(dataset_labels.to_frame())

model = LinearRegression()
model.fit(dataset[["median_income"]], scaled_labels)
some_new_data = dataset[["median_income"]].iloc[:5]  # pretend this is new data

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)


# In[86]:


predictions


# In[87]:


model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
model.fit(dataset[["median_income"]], dataset_labels)
predictions = model.predict(some_new_data)


# In[88]:


predictions


# ## Custom Transformers

# To create simple transformers:

# In[89]:


log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(dataset[["population"]])


# In[90]:


rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[35.0]], gamma=0.1))
age_simil_35 = rbf_transformer.transform(dataset[["dataset_median_age"]])


# In[91]:


age_simil_35


# In[92]:


sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(dataset[["latitude", "longitude"]])


# In[93]:


sf_simil


# In[94]:


ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
ratio_transformer.transform(np.array([[1.0, 2.0], [3.0, 4.0]]))


# In[95]:


# In[96]:


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


# In[97]:


cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)
similarities = cluster_simil.fit_transform(
    dataset[["latitude", "longitude"]], sample_weight=dataset_labels
)


# In[98]:


similarities[:3].round(2)


# In[99]:


# extra code – this cell generates Figure 2–19

dataset_renamed = dataset.rename(
    columns={
        "latitude": "Latitude",
        "longitude": "Longitude",
        "population": "Population",
        "median_house_value": "Median house value (ᴜsᴅ)",
    }
)
dataset_renamed["Max cluster similarity"] = similarities.max(axis=1)

dataset_renamed.plot(
    kind="scatter",
    x="Longitude",
    y="Latitude",
    grid=True,
    s=dataset_renamed["Population"] / 100,
    label="Population",
    c="Max cluster similarity",
    cmap="jet",
    colorbar=True,
    legend=True,
    sharex=False,
    figsize=(10, 7),
)
plt.plot(
    cluster_simil.kmeans_.cluster_centers_[:, 1],
    cluster_simil.kmeans_.cluster_centers_[:, 0],
    linestyle="",
    color="black",
    marker="X",
    markersize=20,
    label="Cluster centers",
)
plt.legend(loc="upper right")
plt.show()


# ## Transformation Pipelines

# Now let's build a pipeline to preprocess the numerical attributes:

# In[100]:


num_pipeline = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler()),
    ]
)


# In[101]:


num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


# In[102]:


from sklearn import set_config

set_config(display="diagram")

num_pipeline


# In[103]:


dataset_num_prepared = num_pipeline.fit_transform(dataset_num)
dataset_num_prepared[:2].round(2)


# In[104]:


def monkey_patch_get_signature_names_out():
    """Monkey patch some classes which did not handle get_feature_names_out()
    correctly in Scikit-Learn 1.0.*."""
    from inspect import Parameter
    from inspect import Signature
    from inspect import signature

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.preprocessing import StandardScaler

    default_get_feature_names_out = StandardScaler.get_feature_names_out

    if not hasattr(SimpleImputer, "get_feature_names_out"):
        print("Monkey-patching SimpleImputer.get_feature_names_out()")
        SimpleImputer.get_feature_names_out = default_get_feature_names_out

    if not hasattr(FunctionTransformer, "get_feature_names_out"):
        print("Monkey-patching FunctionTransformer.get_feature_names_out()")
        orig_init = FunctionTransformer.__init__
        orig_sig = signature(orig_init)

        def __init__(*args, feature_names_out=None, **kwargs):
            orig_sig.bind(*args, **kwargs)
            orig_init(*args, **kwargs)
            args[0].feature_names_out = feature_names_out

        __init__.__signature__ = Signature(
            list(signature(orig_init).parameters.values())
            + [Parameter("feature_names_out", Parameter.KEYWORD_ONLY)]
        )

        def get_feature_names_out(self, names=None):
            if callable(self.feature_names_out):
                return self.feature_names_out(self, names)
            assert self.feature_names_out == "one-to-one"
            return default_get_feature_names_out(self, names)

        FunctionTransformer.__init__ = __init__
        FunctionTransformer.get_feature_names_out = get_feature_names_out


monkey_patch_get_signature_names_out()


# In[105]:


df_dataset_num_prepared = pd.DataFrame(
    dataset_num_prepared, columns=num_pipeline.get_feature_names_out(), index=dataset_num.index
)


# In[106]:


df_dataset_num_prepared.head(2)  # extra code


# In[107]:


num_pipeline.steps


# In[108]:


num_pipeline[1]


# In[109]:


num_pipeline[:-1]


# In[110]:


num_pipeline.named_steps["simpleimputer"]


# In[111]:


num_pipeline.set_params(**{"simpleimputer__strategy": "median"})


# In[112]:


num_attribs = [
    "longitude",
    "latitude",
    "dataset_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer(
    [
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ]
)


# In[113]:


preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)


# In[114]:


dataset_prepared = preprocessing.fit_transform(dataset)


# In[115]:


# extra code – shows that we can get a DataFrame out if we want
dataset_prepared_fr = pd.DataFrame(
    dataset_prepared, columns=preprocessing.get_feature_names_out(), index=dataset.index
)
dataset_prepared_fr.head(2)


# In[116]:


def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )


log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler(),
)
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
preprocessing = ColumnTransformer(
    [
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        (
            "log",
            log_pipeline,
            ["total_bedrooms", "total_rooms", "population", "households", "median_income"],
        ),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline,
)  # one column remaining: dataset_median_age


# In[117]:


dataset_prepared = preprocessing.fit_transform(dataset)
dataset_prepared.shape


# In[118]:


preprocessing.get_feature_names_out()


# # Select and Train a Model

# ## Training and Evaluating on the Training Set

# In[119]:


lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(dataset, dataset_labels)


# Let's try the full preprocessing pipeline on a few training instances:

# In[120]:


dataset_predictions = lin_reg.predict(dataset)
dataset_predictions[:5].round(-2)  # -2 = rounded to the nearest hundred


# Compare against the actual values:

# In[121]:


dataset_labels.iloc[:5].values


# In[122]:


# extra code – computes the error ratios discussed in the book
error_ratios = dataset_predictions[:5].round(-2) / dataset_labels.iloc[:5].values - 1
print(", ".join([f"{100 * ratio:.1f}%" for ratio in error_ratios]))


# In[123]:


lin_rmse = mean_squared_error(dataset_labels, dataset_predictions, squared=False)
lin_rmse


# In[124]:


tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(dataset, dataset_labels)


# In[125]:


dataset_predictions = tree_reg.predict(dataset)
tree_rmse = mean_squared_error(dataset_labels, dataset_predictions, squared=False)
tree_rmse


# ## Better Evaluation Using Cross-Validation

# In[126]:


tree_rmse = -cross_val_score(
    tree_reg, dataset, dataset_labels, scoring="neg_root_mean_squared_error", cv=10
)


# In[127]:


pd.Series(tree_rmse).describe()


# In[128]:


# extra code – computes the error stats for the linear model
lin_rmse = -cross_val_score(
    lin_reg, dataset, dataset_labels, scoring="neg_root_mean_squared_error", cv=10
)
pd.Series(lin_rmse).describe()


# **Warning:** the following cell may take a few minutes to run:

# In[129]:


forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rmse = -cross_val_score(
    forest_reg, dataset, dataset_labels, scoring="neg_root_mean_squared_error", cv=10
)


# In[130]:


pd.Series(forest_rmse).describe()


# Let's compare this RMSE measured using cross-validation
# (the "validation error") with the RMSE measured on the training set
# (the "training error"):

# In[131]:


forest_reg.fit(dataset, dataset_labels)
dataset_predictions = forest_reg.predict(dataset)
forest_rmse = mean_squared_error(dataset_labels, dataset_predictions, squared=False)
forest_rmse


# The training error is much lower than the validation error, which usually
# means that the model has overfit the training set. Another possible
# explanation may be that there's a mismatch between the training data and the
# validation data, but it's not the case here, since both came from the same
# dataset that we shuffled and split in two parts.

# # Fine-Tune Your Model

# ## Grid Search

# **Warning:** the following cell may take a few minutes to run:

# In[132]:


full_pipeline = Pipeline(
    [
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state=42)),
    ]
)
param_grid = [
    {"preprocessing__geo__n_clusters": [5, 8, 10], "random_forest__max_features": [4, 6, 8]},
    {"preprocessing__geo__n_clusters": [10, 15], "random_forest__max_features": [6, 8, 10]},
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error")
grid_search.fit(dataset, dataset_labels)


# You can get the full list of hyperparameters available for tuning by looking
# at `full_pipeline.get_params().keys()`:

# In[133]:


# extra code – shows part of the output of get_params().keys()
print(str(full_pipeline.get_params().keys())[:1000] + "...")


# The best hyperparameter combination found:

# In[134]:


grid_search.best_params_


# In[135]:


grid_search.best_estimator_


# Let's look at the score of each hyperparameter combination tested during the
# grid search:

# In[136]:


cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

# extra code – these few lines of code just make the DataFrame look nicer
cv_res = cv_res[
    [
        "param_preprocessing__geo__n_clusters",
        "param_random_forest__max_features",
        "split0_test_score",
        "split1_test_score",
        "split2_test_score",
        "mean_test_score",
    ]
]
score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
cv_res.columns = ["n_clusters", "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)

cv_res.head()


# ## Randomized Search

# In[137]:


# Try 30 (`n_iter` × `cv`) random combinations of hyperparameters:

# **Warning:** the following cell may take a few minutes to run:

# In[138]:


param_distributions = {
    "preprocessing__geo__n_clusters": randint(low=3, high=50),
    "random_forest__max_features": randint(low=2, high=20),
}

rnd_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
)

rnd_search.fit(dataset, dataset_labels)


# In[139]:


# extra code – displays the random search results
cv_res = pd.DataFrame(rnd_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res = cv_res[
    [
        "param_preprocessing__geo__n_clusters",
        "param_random_forest__max_features",
        "split0_test_score",
        "split1_test_score",
        "split2_test_score",
        "mean_test_score",
    ]
]
cv_res.columns = ["n_clusters", "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)
cv_res.head()


# **Bonus section: how to choose the sampling distribution for a hyperparameter**

# Here are plots of the probability mass functions (for discrete variables),
# and probability density functions (for continuous variables) for `randint()`,
# `uniform()`, `geom()` and `expon()`:


# * `scipy.stats.randint(a, b+1)`: for hyperparameters with _discrete_ values
# that range from a to b, and all values in that range seem equally likely.
xs1 = np.arange(0, 7 + 1)
randint_distribution = randint(0, 7 + 1).pmf(xs1)

# * `scipy.stats.uniform(a, b)`: this is very similar, but for _continuous_
# hyperparameters.
xs2 = np.linspace(0, 7, 500)
uniform_distribution = uniform(0, 7).pdf(xs2)

# * `scipy.stats.geom(1 / scale)`: for discrete values, when you want to sample
# roughly in a given scale. E.g., with scale=1000 most samples will be in this
# ballpark, but ~10% of all samples will be <100 and ~10% will be >2300.
xs3 = np.arange(0, 7 + 1)
geom_distribution = geom(0.5).pmf(xs3)

# * `scipy.stats.expon(scale)`: this is the continuous equivalent of `geom`.
# Just set `scale` to the most likely value.
xs4 = np.linspace(0, 7, 500)
expon_distribution = expon(scale=1).pdf(xs4)

plt.figure(figsize=(12, 7))

plt.subplot(2, 2, 1)
plt.bar(xs1, randint_distribution, label="scipy.randint(0, 7 + 1)")
plt.ylabel("Probability")
plt.legend()
plt.axis([-1, 8, 0, 0.2])

plt.subplot(2, 2, 2)
plt.fill_between(xs2, uniform_distribution, label="scipy.uniform(0, 7)")
plt.ylabel("PDF")
plt.legend()
plt.axis([-1, 8, 0, 0.2])

plt.subplot(2, 2, 3)
plt.bar(xs3, geom_distribution, label="scipy.geom(0.5)")
plt.xlabel("Hyperparameter value")
plt.ylabel("Probability")
plt.legend()
plt.axis([0, 7, 0, 1])

plt.subplot(2, 2, 4)
plt.fill_between(xs4, expon_distribution, label="scipy.expon(scale=1)")
plt.xlabel("Hyperparameter value")
plt.ylabel("PDF")
plt.legend()
plt.axis([0, 7, 0, 1])

plt.show()

# * `scipy.stats.loguniform(a, b)`: when you have almost no idea what the
# optimal hyperparameter value's scale is. If you set a=0.01 and b=100, then
# you're just as likely to sample a value between 0.01 and 0.1 as a value
# between 10 and 100.

# Here are the PDF for `expon()` and `loguniform()` (left column), as well as
# the PDF of log(X) (right column). The right column shows the distribution of
# hyperparameter _scales_. You can see that `expon()` favors hyperparameters
# with roughly the desired scale, with a longer tail towards the smaller
# scales. But `loguniform()` does not favor any scale, they are all equally
# likely:


# extra code – shows the difference between expon and loguniform


xs1 = np.linspace(0, 7, 500)
expon_distribution = expon(scale=1).pdf(xs1)

log_xs2 = np.linspace(-5, 3, 500)
log_expon_distribution = np.exp(log_xs2 - np.exp(log_xs2))

xs3 = np.linspace(0.001, 1000, 500)
loguniform_distribution = loguniform(0.001, 1000).pdf(xs3)

log_xs4 = np.linspace(np.log(0.001), np.log(1000), 500)
log_loguniform_distribution = uniform(np.log(0.001), np.log(1000)).pdf(log_xs4)

plt.figure(figsize=(12, 7))

plt.subplot(2, 2, 1)
plt.fill_between(xs1, expon_distribution, label="scipy.expon(scale=1)")
plt.ylabel("PDF")
plt.legend()
plt.axis([0, 7, 0, 1])

plt.subplot(2, 2, 2)
plt.fill_between(log_xs2, log_expon_distribution, label="log(X) with X ~ expon")
plt.legend()
plt.axis([-5, 3, 0, 1])

plt.subplot(2, 2, 3)
plt.fill_between(xs3, loguniform_distribution, label="scipy.loguniform(0.001, 1000)")
plt.xlabel("Hyperparameter value")
plt.ylabel("PDF")
plt.legend()
plt.axis([0.001, 1000, 0, 0.005])

plt.subplot(2, 2, 4)
plt.fill_between(log_xs4, log_loguniform_distribution, label="log(X) with X ~ loguniform")
plt.xlabel("Log of hyperparameter value")
plt.legend()
plt.axis([-8, 1, 0, 0.2])

plt.show()


# ## Analyze the Best Models and Their Errors

# In[142]:


final_model = rnd_search.best_estimator_  # includes preprocessing
feature_importances = final_model["random_forest"].feature_importances_
feature_importances.round(2)


# In[143]:


sorted(zip(feature_importances, final_model["preprocessing"].get_feature_names_out()), reverse=True)


# ## Evaluate Your System on the Test Set

# In[144]:


X_test = stratified_test_set.drop("median_house_value", axis=1)
y_test = stratified_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)

final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
print(final_rmse)


# We can compute a 95% confidence interval for the test RMSE:

# In[145]:


confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(
    stats.t.interval(
        confidence,
        len(squared_errors) - 1,
        loc=squared_errors.mean(),
        scale=stats.sem(squared_errors),
    )
)


# We could compute the interval manually like this:

# In[146]:


# extra code – shows how to compute a confidence interval for the RMSE
m = len(squared_errors)
mean = squared_errors.mean()
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)


# Alternatively, we could use a z-score rather than a t-score. Since the test
# set is not too small, it won't make a big difference:

# In[147]:


# extra code – computes a confidence interval again using a z-score
zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)


# ## Model persistence using joblib

# Save the final model:

# In[148]:


joblib.dump(final_model, "my_california_dataset_model.pkl")


# Now you can deploy this model to production. For example, the following code
# could be a script that would run in production:

# In[149]:


# extra code – excluded for conciseness


# class ClusterSimilarity(BaseEstimator, TransformerMixin):
#    [...]

final_model_reloaded = joblib.load("my_california_dataset_model.pkl")

new_data = dataset.iloc[:5]  # pretend these are new districts
predictions = final_model_reloaded.predict(new_data)


# In[150]:


predictions


# You could use pickle instead, but joblib is more efficient.

# # Exercise solutions

# ## 1.

# Exercise: _Try a Support Vector Machine regressor (`sklearn.svm.SVR`) with
# various hyperparameters, such as `kernel="linear"` (with various values for
# the `C` hyperparameter) or `kernel="rbf"` (with various values for the `C`
# and `gamma` hyperparameters). Note that SVMs don't scale well to large
# datasets, so you should probably train your model on just the first 5,000
# instances of the training set and use only 3-fold cross-validation, or else
# it will take hours. Don't worry about what the hyperparameters mean for now
# (see the SVM notebook if you're interested). How does the best `SVR`
# predictor perform?_

# In[151]:


param_grid = [
    {
        "svr__kernel": ["linear"],
        "svr__C": [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0],
    },
    {
        "svr__kernel": ["rbf"],
        "svr__C": [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
        "svr__gamma": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    },
]

svr_pipeline = Pipeline([("preprocessing", preprocessing), ("svr", SVR())])
grid_search = GridSearchCV(svr_pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error")
grid_search.fit(dataset.iloc[:5000], dataset_labels.iloc[:5000])


# The best model achieves the following score (evaluated using 3-fold cross validation):

# In[152]:


svr_grid_search_rmse = -grid_search.best_score_
svr_grid_search_rmse


# That's much worse than the `RandomForestRegressor` (but to be fair, we
# trained the model on much less data). Let's check the best hyperparameters
# found:

# In[153]:


grid_search.best_params_


# The linear kernel seems better than the RBF kernel. Notice that the value of
# `C` is the maximum tested value. When this happens you definitely want to
# launch the grid search again with higher values for `C` (removing the
# smallest values), because it is likely that higher values of `C` will be
# better.

# ## 2.

# Exercise: _Try replacing the `GridSearchCV` with a `RandomizedSearchCV`._

# **Warning:** the following cell will take several minutes to run. You can
# specify `verbose=2` when creating the `RandomizedSearchCV` if you want to see
# the training details.

# In[154]:


# see https://docs.scipy.org/doc/scipy/reference/stats.html
# for `expon()` and `loguniform()` documentation and more probability distribution functions.

# Note: gamma is ignored when kernel is "linear"
param_distributions = {
    "svr__kernel": ["linear", "rbf"],
    "svr__C": loguniform(20, 200_000),
    "svr__gamma": expon(scale=1.0),
}

rnd_search = RandomizedSearchCV(
    svr_pipeline,
    param_distributions=param_distributions,
    n_iter=50,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
)
rnd_search.fit(dataset.iloc[:5000], dataset_labels.iloc[:5000])


# The best model achieves the following score (evaluated using 3-fold cross validation):

# In[155]:


svr_rnd_search_rmse = -rnd_search.best_score_
svr_rnd_search_rmse


# Now that's really much better, but still far from the
# `RandomForestRegressor`'s performance. Let's check the best hyperparameters
# found:

# In[156]:


rnd_search.best_params_


# This time the search found a good set of hyperparameters for the RBF kernel.
# Randomized search tends to find better hyperparameters than grid search in
# the same amount of time.

# Note that we used the `expon()` distribution for `gamma`, with a scale of 1,
# so `RandomSearch` mostly searched for values roughly of that scale: about 80%
# of the samples were between 0.1 and 2.3 (roughly 10% were smaller and 10%
# were larger):

# In[157]:


np.random.seed(42)

s = expon(scale=1).rvs(100_000)  # get 100,000 samples
((s > 0.105) & (s < 2.29)).sum() / 100_000


# We used the `loguniform()` distribution for `C`, meaning we did not have a
# clue what the optimal scale of `C` was before running the random search. It
# explored the range from 20 to 200 just as much as the range from 2,000 to
# 20,000 or from 20,000 to 200,000.

# ## 3.

# Exercise: _Try adding a `SelectFromModel` transformer in the preparation
# pipeline to select only the most important attributes._

# Let's create a new pipeline that runs the previously defined preparation
# pipeline, and adds a `SelectFromModel` transformer based on a
# `RandomForestRegressor` before the final regressor:

# In[158]:


selector_pipeline = Pipeline(
    [
        ("preprocessing", preprocessing),
        (
            "selector",
            SelectFromModel(RandomForestRegressor(random_state=42), threshold=0.005),
        ),  # min feature importance
        (
            "svr",
            SVR(
                C=rnd_search.best_params_["svr__C"],
                gamma=rnd_search.best_params_["svr__gamma"],
                kernel=rnd_search.best_params_["svr__kernel"],
            ),
        ),
    ]
)


# In[159]:


selector_rmse = -cross_val_score(
    selector_pipeline,
    dataset.iloc[:5000],
    dataset_labels.iloc[:5000],
    scoring="neg_root_mean_squared_error",
    cv=3,
)
pd.Series(selector_rmse).describe()


# Oh well, feature selection does not seem to help. But maybe that's just
# because the threshold we used was not optimal. Perhaps try tuning it using
# random search or grid search?

# ## 4.

# Exercise: _Try creating a custom transformer that trains a
# k-Nearest Neighbors regressor (`sklearn.neighbors.KNeighborsRegressor`) in
# its `fit()` method, and outputs the model's predictions in its `transform()`
# method. Then add this feature to the preprocessing pipeline, using latitude
# and longitude as the inputs to this transformer. This will add a feature in
# the model that corresponds to the dataset median price of the nearest
# districts._

# Rather than restrict ourselves to k-Nearest Neighbors regressors, let's
# create a transformer that accepts any regressor. For this, we can extend the
# `MetaEstimatorMixin` and have a required `estimator` argument in the
# constructor. The `fit()` method must work on a clone of this estimator, and
# it must also save `feature_names_in_`. The `MetaEstimatorMixin` will ensure
# that `estimator` is listed as a required parameters, and it will update
# `get_params()` and `set_params()` to make the estimator's hyperparameters
# available for tuning. Lastly, we create a `get_feature_names_out()` method:
# the output column name is the ...

# In[160]:


# Let's ensure it complies to Scikit-Learn's API:

# In[161]:


check_estimator(FeatureFromRegressor(KNeighborsRegressor()))


# Good! Now let's test it:

# In[162]:


knn_reg = KNeighborsRegressor(n_neighbors=3, weights="distance")
knn_transformer = FeatureFromRegressor(knn_reg)
geo_features = dataset[["latitude", "longitude"]]
knn_transformer.fit_transform(geo_features, dataset_labels)


# And what does its output feature name look like?

# In[163]:


knn_transformer.get_feature_names_out()


# Okay, now let's include this transformer in our preprocessing pipeline:

# In[164]:


transformers = [
    (name, clone(transformer), columns) for name, transformer, columns in preprocessing.transformers
]
geo_index = [name for name, _, _ in transformers].index("geo")
transformers[geo_index] = ("geo", knn_transformer, ["latitude", "longitude"])

new_geo_preprocessing = ColumnTransformer(transformers)


# In[165]:


new_geo_pipeline = Pipeline(
    [
        ("preprocessing", new_geo_preprocessing),
        (
            "svr",
            SVR(
                C=rnd_search.best_params_["svr__C"],
                gamma=rnd_search.best_params_["svr__gamma"],
                kernel=rnd_search.best_params_["svr__kernel"],
            ),
        ),
    ]
)


# In[166]:


new_pipe_rmse = -cross_val_score(
    new_geo_pipeline,
    dataset.iloc[:5000],
    dataset_labels.iloc[:5000],
    scoring="neg_root_mean_squared_error",
    cv=3,
)
pd.Series(new_pipe_rmse).describe()


# Yikes, that's terrible! Apparently the cluster similarity features were much
# better. But perhaps we should tune the `KNeighborsRegressor`'s
# hyperparameters? That's what the next exercise is about.

# ## 5.

# Exercise: _Automatically explore some preparation options using `RandomSearchCV`._

# In[167]:


param_distributions = {
    "preprocessing__geo__estimator__n_neighbors": range(1, 30),
    "preprocessing__geo__estimator__weights": ["distance", "uniform"],
    "svr__C": loguniform(20, 200_000),
    "svr__gamma": expon(scale=1.0),
}

new_geo_rnd_search = RandomizedSearchCV(
    new_geo_pipeline,
    param_distributions=param_distributions,
    n_iter=50,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
)
new_geo_rnd_search.fit(dataset.iloc[:5000], dataset_labels.iloc[:5000])


# In[168]:


new_geo_rnd_search_rmse = -new_geo_rnd_search.best_score_
new_geo_rnd_search_rmse


# Oh well... at least we tried! It looks like the cluster similarity features
# are definitely better than the KNN feature. But perhaps you could try having
# both? And maybe training on the full training set would help as well.

# ## 6.

# Exercise: _Try to implement the `StandardScalerClone` class again from
# scratch, then add support for the `inverse_transform()` method: executing
# `scaler.inverse_transform(scaler.fit_transform(X))` should return an array
# very close to `X`. Then add support for feature names: set `feature_names_in_
# ` in the `fit()` method if the input is a DataFrame. This attribute should be
# a NumPy array of column names. Lastly, implement the `get_feature_names_out()`
# method: it should have one optional `input_features=None` argument. If
# passed, the method should check that its length matches `n_features_in_`, and
# it should match `feature_names_in_` if it is defined, then `input_features`
# should be returned. If `input_features` is `None`, then the method should
# return `feature_names_in_` if it is defined or `np.array(["x0", "x1", ...])`
# with length `n_features_in_` otherwise._

# In[169]:


# Let's test our custom transformer:

# In[170]:


check_estimator(StandardScalerClone())


# No errors, that's a great start, we respect the Scikit-Learn API.

# Now let's ensure the transformation works as expected:

# In[171]:


np.random.seed(42)
X = np.random.rand(1000, 3)

scaler = StandardScalerClone()
X_scaled = scaler.fit_transform(X)

assert np.allclose(X_scaled, (X - X.mean(axis=0)) / X.std(axis=0))


# How about setting `with_mean=False`?

# In[172]:


scaler = StandardScalerClone(with_mean=False)
X_scaled_un_centered = scaler.fit_transform(X)

assert np.allclose(X_scaled_un_centered, X / X.std(axis=0))


# And does the inverse work?

# In[173]:


scaler = StandardScalerClone()
X_back = scaler.inverse_transform(scaler.fit_transform(X))

assert np.allclose(X, X_back)


# How about the feature names out?

# In[174]:


assert np.all(scaler.get_feature_names_out() == ["x0", "x1", "x2"])
assert np.all(scaler.get_feature_names_out(["a", "b", "c"]) == ["a", "b", "c"])


# And if we fit a DataFrame, are the feature in and out ok?

# In[175]:


df = pd.DataFrame({"a": np.random.rand(100), "b": np.random.rand(100)})
scaler = StandardScalerClone()
X_scaled = scaler.fit_transform(df)

assert np.all(scaler.feature_names_in_ == ["a", "b"])
assert np.all(scaler.get_feature_names_out() == ["a", "b"])


# All good! That's all for today! 😀

# Congratulations! You already know quite a lot about Machine Learning. :)
