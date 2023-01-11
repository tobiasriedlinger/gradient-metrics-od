from sklearn.metrics import accuracy_score, roc_auc_score, r2_score


def evaluate_classifier_performance(y_test, y_pred):
    return accuracy_score(y_test, y_pred >= 0.5), roc_auc_score(y_test, y_pred)


def evaluate_regression_performance(y_test, y_pred):
    return r2_score(y_test, y_pred)
