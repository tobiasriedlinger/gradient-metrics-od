import sklearn.linear_model as sklin


def linear_regression_from_parameter_dict(params):
    return sklin.LinearRegression(**params)


def linear_regression_parameter_selection():
    return sklin.LinearRegression()
