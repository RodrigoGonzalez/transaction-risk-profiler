""" Scaling transformer """
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.scale_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray, y=None) -> StandardScalerClone:
        """Learn the mean and standard deviation of X."""
        _ = y  # unused
        X_orig = X
        # checks that X is an array with finite float values
        X: np.ndarray = check_array(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        # every estimator stores this in fit()
        self.n_features_in_ = X.shape[1]
        if hasattr(X_orig, "columns"):
            self.feature_names_in_ = np.array(X_orig.columns, dtype=object)
        return self  # always return self!

    def transform(self, X):
        """Scale features of X to zero mean and unit variance"""
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X: np.ndarray = check_array(X)
        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                f"Unexpected number of features: {self.n_features_in_} != {X.shape[1]}"
            )
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features of X back to the original distribution"""
        check_is_fitted(self)
        X: np.ndarray = check_array(X)
        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                f"Unexpected number of features: {self.n_features_in_} != {X.shape[1]}"
            )
        X = X * self.scale_
        return X + self.mean_ if self.with_mean else X

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features"""
        if input_features is None:
            return getattr(self, "feature_names_in_", [f"x{i}" for i in range(self.n_features_in_)])
        if len(input_features) != self.n_features_in_:
            raise ValueError(
                f"Invalid number of features: {len(input_features)} != {self.n_features_in_}"
            )
        if hasattr(self, "feature_names_in_") and not np.all(
            self.feature_names_in_ == input_features
        ):
            raise ValueError(
                f"input_features: {input_features} ≠ feature_names_in_: {self.feature_names_in_}"
            )
        return input_features
