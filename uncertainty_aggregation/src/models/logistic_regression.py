import sklearn.linear_model as sklin


def logistic_regression_from_parameter_dict(params):
    return sklin.LogisticRegression(**params)


def logistic_regression_parameter_selection():
    return sklin.LogisticRegression()
