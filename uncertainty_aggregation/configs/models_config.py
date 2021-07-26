import src.models.gradient_boosting as gb
import src.models.logistic_regression as logreg
import src.models.linear_regression as linreg
import sklearn.linear_model as sklin

CLASSIFICATION_MODELS = ["logistic", "gb_classifier"]
REGRESSION_MODELS = ["linear_regression", "gb_regression"]

GB_CLASSIFIER_PARAMETERS = {"n_estimators" : list(range(10, 30)),
                        "max_depth" : list(range(2, 8)),
                        "learning_rate" : [0.3],
                        "reg_alpha" : [0.5, 1.0, 1.5],
                        "reg_lambda" : [0.0],
                        "use_label_encoder": [False]}

GB_REGRESSION_PARAMETERS = {"n_estimators" : list(range(10, 20)),
                            "max_depth" : list(range(2, 4)),
                            "learning_rate" : [0.3],
                            "reg_alpha" : [0.5, 1.0, 2.0],
                            "reg_lambda" : [0.0, 0.5]}

LOGISTIC_REGRESSION_PARAMETERS = {"penalty" : ["l2"],
                                  "C" : [0.5, 0.3, 0.1, 0.05, 0.01],
                                  "solver" : ["saga"],
                                  "max_iter" : [5000]}

LINEAR_REGRESSION_PARAMETERS = {"fit_intercept" : [True]}

PARAMETER_SEARCH_MODELS = {"gb_classifier" : gb.gb_classifier_parameter_selection,
                           "logistic" : logreg.logistic_regression_parameter_selection,
                           "gb_regression" : gb.gb_regression_parameter_selection,
                           "linear_regression" : sklin.LinearRegression}
PARAMETER_SEARCH_OPTIONS = {"gb_classifier" : GB_CLASSIFIER_PARAMETERS,
                            "logistic" : LOGISTIC_REGRESSION_PARAMETERS,
                            "gb_regression" : GB_REGRESSION_PARAMETERS,
                            "linear_regression" : LINEAR_REGRESSION_PARAMETERS}

get_model_from_parameter_dict = {"gb_classifier" : gb.gb_classifier_from_parameter_dict,
                                 "logistic" : logreg.logistic_regression_from_parameter_dict,
                                 "gb_regression" : gb.gb_regression_from_parameter_dict,
                                 "linear_regression" : linreg.linear_regression_from_parameter_dict}

# DEFAULT_PARAMETER_SEARCH_SCORE_THRESHOLDS = [1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5]
DEFAULT_PARAMETER_SEARCH_SCORE_THRESHOLDS = [0.5]
