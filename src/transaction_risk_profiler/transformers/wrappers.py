"""Module for wrappers of Scikit-learn pre-processing transformers. These wrappers allow
Scikit-learn pipelines to use the transformer.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class ColumnTransformerWrapper(BaseEstimator, TransformerMixin):
    """A wrapper for Scikit-learn transformers.

    This class allows Scikit-learn transformers to be used in pipelines.

    Attributes
    ----------
    variables : list
        The variables to be transformed.
    transformer : object
        The Scikit-learn transformer to be used.
    """

    def __init__(self, variables: list[str] | str | None = None, transformer=None):
        """Initialize the transformer wrapper.

        Parameters
        ----------
        variables : list, optional
            The variables to be transformed. If not a list, it will be
            converted into a list.
        transformer : object, optional
            The Scikit-learn transformer to be used.
        """
        self.variables = variables if isinstance(variables, list) else [variables]
        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> ColumnTransformerWrapper:
        """Fit the transformer to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to fit the transformer to.
        y : pd.Series, optional
            The target variable. Not used, present for API consistency.

        Returns
        -------
        self : ColumnTransformerWrapper
            The instance itself.
        """
        self.transformer.fit(X[self.variables])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to be transformed.

        Returns
        -------
        X : pd.DataFrame
            The transformed data.
        """
        X = X.copy()
        X[self.variables] = self.transformer.transform(X[self.variables])
        return X


class FunctionTransformer(BaseEstimator, TransformerMixin):
    """A transformer for applying a function to the data.

    This class allows a function to be used as a transformer in a pipeline.

    Attributes
    ----------
    func : callable
        The function to be applied.
    """

    def __init__(self, func: callable):
        """Initialize the function transformer.

        Parameters
        ----------
        func : callable
            The function to be applied.
        """
        self.func = func

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> FunctionTransformer:
        """Fit the transformer to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to fit the transformer to.
        y : pd.Series, optional
            The target variable. Not used, present for API consistency.

        Returns
        -------
        self : FunctionTransformer
            The instance itself.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to be transformed.

        Returns
        -------
        X : pd.DataFrame
            The transformed data.
        """
        X = X.copy()
        X = self.func(X)
        return X


class TemporalVariableEstimator(BaseEstimator, TransformerMixin):
    """A transformer for temporal variables.

    This class allows the transformation of temporal variables in a DataFrame.

    Attributes
    ----------
    variables : list
        The variables to be transformed.
    reference_variables : str
        The reference variable for the transformation.
    """

    def __init__(self, variables: list[str] | str | None = None, reference_variable: str = None):
        """Initialize the temporal variable estimator.

        Parameters
        ----------
        variables : list, optional
            The variables to be transformed. If not a list, it will be
            converted into a list.
        reference_variable : str, optional
            The reference variable for the transformation.
        """
        self.variables = variables if isinstance(variables, list) else [variables]
        self.reference_variables = reference_variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> TemporalVariableEstimator:
        """Fit the estimator to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to fit the estimator to.
        y : pd.Series, optional
            The target variable. Not used, present for API consistency.

        Returns
        -------
        self : TemporalVariableEstimator
            The instance itself.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to be transformed.

        Returns
        -------
        X : pd.DataFrame
            The transformed data.
        """
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variables] - X[feature]

        return X


class DropFeatures(BaseEstimator, TransformerMixin):
    """A transformer for dropping features.

    This class allows the dropping of specified features in a DataFrame.

    Attributes
    ----------
    variables : list
        The variables to be dropped.
    """

    def __init__(self, variables_to_drop: list[str] | None = None):
        """Initialize the feature dropper.

        Parameters
        ----------
        variables_to_drop : list, optional
            The variables to be dropped.
        """
        self.variables = variables_to_drop

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> DropFeatures:
        """Fit the dropper to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to fit the dropper to.
        y : pd.Series, optional
            The target variable. Not used, present for API consistency.

        Returns
        -------
        self : DropFeatures
            The instance itself.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to be transformed.

        Returns
        -------
        X : pd.DataFrame
            The transformed data.
        """
        X = X.copy()
        X = X.drop(self.variables, axis=1)
        return X
