import xgboost


def initialize_new_gb_classifier(n_estimators, max_depth, learning_rate, reg_alpha, reg_lambda, colsample_bytree=0.5):
    model = xgboost.XGBClassifier(verbosity=1, max_depth=max_depth, colsample_bytree=colsample_bytree,
                                  n_estimators=n_estimators, learning_rate=learning_rate, reg_alpha=reg_alpha, reg_lambda=reg_lambda)
    return model


def load_gb_classifier(file_path):
    model = xgboost.XGBClassifier()
    model.load_model(file_path)
    return model


def gb_classifier_from_parameter_dict(params):
    return xgboost.XGBClassifier(tree_method="gpu_hist", gpu_id=0, **params)


def gb_classifier_parameter_selection(use_gpu=True):
    options = {True: {"tree_method": "gpu_hist", "gpu_id": 0}, False: {}}
    return xgboost.XGBClassifier(**options[use_gpu])

#################################################


def initialize_new_gb_regression(n_estimators, max_depth, learning_rate, reg_alpha, reg_lambda, colsample_bytree=0.5):
    model = xgboost.XGBRegressor(verbosity=1, max_depth=max_depth, colsample_bytree=colsample_bytree,
                                 n_estimators=n_estimators, learning_rate=learning_rate, reg_alpha=reg_alpha, reg_lambda=reg_lambda)
    return model


def load_gb_regression(file_path):
    model = xgboost.XGBRegressor()
    model.load_model(file_path)
    return model


def gb_regression_from_parameter_dict(params):
    return xgboost.XGBRegressor(tree_method="gpu_hist", gpu_id=0, **params)


def gb_regression_parameter_selection(use_gpu=True):
    options = {True: {"tree_method": "gpu_hist", "gpu_id": 0}, False: {}}
    return xgboost.XGBRegressor(**options[use_gpu])
