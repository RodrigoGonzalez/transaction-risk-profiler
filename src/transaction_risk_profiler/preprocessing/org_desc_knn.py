import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

from transaction_risk_profiler.utils.text_utils import extract_text

logger = logging.getLogger(__name__)
df = pd.read_json("data/train_new.json")
tfv = df[["object_id", "acct_type", "org_desc"]]
del df
tfv["org_desc"] = tfv["org_desc"].apply(lambda x: x.encode("utf-8") if x is not None else "")
tfv["org_desc"] = tfv["org_desc"].apply(extract_text)
tfv["acct_type"] = tfv["acct_type"].apply(lambda x: 0 if x == "premium" else 1)
tfv_no_blanks = tfv[tfv["org_desc"] != ""]
labels = tfv_no_blanks["org_desc"]
logger.info("done processing shit!!!")

vectorizer = TfidfVectorizer(stop_words="english", min_df=3, norm="l2")
X = vectorizer.fit_transform(tfv_no_blanks["org_desc"])
X_all = vectorizer.fit_transform(tfv["org_desc"])
logger.info("done vectorizing shit!!!")

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X, labels)
logger.info("done with knn!")

# CHECK IN THE FINAL MODEL HOW FRAUD IS LABELED, 1 OR 0!!!
# HERE, FRAUD IS LABELED AS 1
neighbors = knn.kneighbors(X_all)[1]
logger.info("done with knn.kneighbors shit!!!")
tfv["org_desc_knn"] = 0.0
for counter, i in enumerate(range(len(tfv)), start=1):
    tfv["org_desc_knn"][i] = tfv["acct_type"][neighbors[i]].mean()
    logger.info(counter)