""" Preprocessing the data """
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataPreProcessing:
    """
    Preprocessing the data
    """

    def __init__(self, file_path: str) -> None:
        self.file_name = file_path
        self.df = None
        self._data_load()

    @staticmethod
    def _average_previous_payouts(dict_list: list) -> float:
        """
        Returns the average payout amount from previous_payouts column

        Parameters
        ----------
        dict_list : list
            A list of dictionaries.

        Returns
        -------
        float
            The average payout amount from previous_payouts column.
        """
        if not dict_list:
            return 0
        amount = sum(dic["amount"] or 0 for dic in dict_list)
        return float(amount) / len(dict_list)

    def _data_load(self) -> None:
        """Load the data"""
        self.df = pd.read_json(self.file_name)
        self._data_cleaning()
        self._feature_engineering()

    @staticmethod
    def _get_feature(func: callable, feat: str, x: list) -> float:
        """
        input: function, string, list of dictionary's
        output: float

        Returns the function value applied to the input feature
        """
        val = [i[feat] for i in x]
        return func(val) if val else 0

    def _data_cleaning(self) -> None:
        """
        Performs simple data cleaning.
        """
        self.df["venue_name"] = self.df["venue_name"].apply(lambda x: "" if x == "null" else x)
        self.df["channels"] = self.df["channels"].astype(str)
        self.df["has_org_name"] = self.df["org_name"].apply(lambda x: "1" if x else "0")
        self.df["has_org_desc"] = self.df["org_desc"].apply(lambda x: "1" if x else "0")
        self.df["has_payee"] = self.df["payee_name"].apply(lambda x: "1" if x else "0")
        col_cat = ["has_analytics", "has_logo", "show_map", "user_type"]
        for col in col_cat:
            self.df[col] = self.df[col].astype(str)
        self.df.remove_columns(
            [
                "object_id",
                "venue_country",
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

    def _feature_engineering(self) -> None:
        """
        Performs feature engineering.
        """
        self.df["num_previous_payouts"] = self.df["previous_payouts"].apply(lambda x: len(x))
        self.df["create_lag"] = self.df["event_created"] - self.df["user_created"]
        self.df["start_lag"] = self.df["event_start"] - self.df["event_created"]
        self.df["payout_lag"] = self.df["approx_payout_date"] - self.df["event_created"]
        self.df["start_payout_lag"] = self.df["event_start"] - self.df["approx_payout_date"]
        self.df["venue_name_length"] = self.df["venue_name"].apply(lambda x: len(x))
        self.df["org_name_length"] = self.df["org_name"].apply(lambda x: len(x))
        self.df["description_length"] = self.df["description"].apply(lambda x: len(x))
        self.df["avg_previous_payouts"] = self.df["previous_payouts"].apply(
            lambda x: self._average_previous_payouts(x)
        )
        extractions = [
            ("max_cost", (max, "cost")),
            ("min_cost", (min, "cost")),
            ("mean_cost", (np.mean, "cost")),
            ("std_cost", (np.std, "cost")),
            ("tickets_sold", (sum, "quantity_sold")),
            ("total_tickets", (sum, "quantity_total")),
        ]
        for extract in extractions:
            col_name = extract[0]
            self.df[col_name] = self.df["ticket_types"].apply(
                lambda x: self._get_feature(extract[1][0], extract[1][1], x)
            )


if __name__ == "__main__":
    # path = "test.json"
    path = "data/train_new.json"
    data = DataPreProcessing(path)
    logger.info(data.df.head())

# train, test = sf.random_split(0.8, seed=25)
# boosted_tree = gl.boosted_tree_classifier.create(train, "acct_type")
# boosted_tree.save("boosted_tree")


# """
# # ticket type extraction
# def get_feature(func, feat, dict_list):
#     return func([i[feat] for i in dict_list])
# apply to sf2[ticket_types]:
# num of ticket types = len(dict_list)
# max cost sf2['ticket_types'].apply(lambda dict_list: get_feature(max, 'cost', dict_list))
#
# extractions = [
#     ("max_cost", (max, "cost")),
#     ("min_cost", (min, "cost")),
#     ("mean_cost", (np.mean, "cost")),
#     ("std_cost", (np.std, "cost")),
#     ("tickets_sold", (sum, "quantity_sold")),
#     ("total_tickets", (sum, "quantity_total")),
# ]
#
# for extract in extractions:
#     col_name = extract[0]
#     sf2[col_name] = sf2["ticket_types"].apply(
#         lambda dict_list: get_feature(extract[1][0], extract[1][1], dict_list)
#     )
# """
