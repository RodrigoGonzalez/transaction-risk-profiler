from time import time

from sklearn import metrics
from sklearn.utils.extmath import density

from transaction_risk_profiler.modeling.baseline import X_test
from transaction_risk_profiler.modeling.baseline import X_train
from transaction_risk_profiler.modeling.baseline import logger
from transaction_risk_profiler.modeling.baseline import y_test
from transaction_risk_profiler.modeling.baseline import y_train


def benchmark_model(clf):
    logger.info("_" * 80)
    logger.info("Training: ")
    logger.info(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    logger.info("train time: %0.3fs" % train_time)

    t0 = time()
    y_pred = clf.predict(X_test)
    test_time = time() - t0
    logger.info("test time: %0.3fs" % test_time)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    logger.info("accuracy:   %0.3f" % accuracy)
    logger.info("recall: %0.3f" % recall)
    logger.info("precision: %0.3f" % precision)

    if hasattr(clf, "coef_"):
        logger.info("dimensionality: %d" % clf.coef_.shape[1])
        logger.info("density: %f" % density(clf.coef_))

    logger.info()
    clf_descr = str(clf).split("(")[0]
    return clf_descr, accuracy, recall, precision, train_time, test_time
