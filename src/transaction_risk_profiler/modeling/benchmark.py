"""
This module benchmarks the performance of models for transaction risk profiling.
It measures the training and testing time, and evaluates the models based on
accuracy, recall, and precision metrics.
"""
import logging
import timeit

import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import density

logger = logging.getLogger(__name__)


def benchmark_model(
    clf: BaseEstimator,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[str, float, float, float, float, float]:
    """
    Benchmark the given model by measuring its training and testing time, and
    evaluating its accuracy, recall, and precision.

    Parameters
    ----------
    clf : BaseEstimator
        The model to be benchmarked.
    X_train : np.ndarray
        Training data.
    X_test : np.ndarray
        Testing data.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Testing labels.

    Returns
    -------
    Tuple[str, float, float, float, float, float]
        A tuple containing the model name, accuracy, recall, precision,
        training time, and testing time.
    """
    logger.info("_" * 80)
    logger.info(f"Starting training for model: {clf}")
    train_time = timeit(lambda: clf.fit(X_train, y_train), number=1)
    logger.info(f"Training completed. Time taken: {train_time:.3f}s")

    logger.info(f"Starting prediction for model: {clf}")
    y_pred = clf.predict(X_test)
    test_time = timeit(lambda: clf.predict(X_test), number=1)
    logger.info(f"Prediction completed. Time taken: {test_time:.3f}s")

    accuracy = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    logger.info(
        f"Model performance metrics: Accuracy: {accuracy:.3f}, Recall: "
        f"{recall:.3f}, Precision: {precision:.3f}"
    )

    if hasattr(clf, "coef_"):
        logger.info(f"Model dimensionality: {clf.coef_.shape[1]}")
        logger.info(f"Model density: {density(clf.coef_)}")

    logger.info(f"Benchmarking completed for model: {clf}")
    clf_descr = str(clf).split("(")[0]
    return clf_descr, accuracy, recall, precision, train_time, test_time
