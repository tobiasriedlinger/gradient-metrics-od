df_path = "/home/riedlinger/MetaDetect-TestEvaluation/grads_lr=1e-3_bs=64"

gt_path = "/home/riedlinger/datasets_ground_truth/KITTI/csv"

num_classes = 8

# Set of uncertainty metrics to use. Should contain "score" xor "output"
"""
Options: 
    - "score" / "output"
    - "meta_detect"
    - "gradient_metrics"
"""
# metrics_constellation = ["score"]
# metrics_constellation = ["output"]
# metrics_constellation = ["output", "meta_detect", "gradient_metrics"]
# metrics_constellation = ["score", "meta_detect"]
# metrics_constellation = ["score", "gradient_metrics"]
# metrics_constellation = ["output", "gradient_metrics"]
metrics_constellation = ["gradient_metrics"]

"""
Options:
    - "logistic_regression"
    - "gb_classifier"
    - "linear_regression"
    - "gb_regression"
"""
aggregation_model = "gb_classifier"
# aggregation_model = "gb_regression"

# CLASSIFICATION_MODELS = ["logistic_regression", "gb_classifier"]
# REGRESSION_MODELS = ["linear_regression", "gb_regression"]

OUTPUT_METRICS = ["xmin", "ymin", "xmax", "ymax", "s", "category_idx",
                  "prob_sum"] + [f"prob_{i}" for i in range(num_classes)]
STD_OUTPUT_METRICS = ["s", "prob_sum"] + \
    [f"prob_{i}" for i in range(num_classes)]
META_DETECT_METRICS = ['Number of Candidate Boxes', 'x_min',
                       'x_max', 'x_mean', 'x_std', 'y_min', 'y_max', 'y_mean', 'y_std', 'w_min', 'w_max', 'w_mean', 'w_std',
                       'h_min', 'h_max', 'h_mean', 'h_std', 'size', 'size_min', 'size_max', 'size_mean', 'size_std',
                       'circum', 'circum_min', 'circum_max', 'circum_mean', 'circum_std', 'size/circum', 'size/circum_min',
                       'size/circum_max', 'size/circum_mean', 'size/circum_std', 'score_min', 'score_mean', 'score_std',
                       'IoU_pb_min', 'IoU_pb_max', 'IoU_pb_mean', 'IoU_pb_std']
PERFORM_PARAMETER_SEARCH = False

# PARAMETER_SEARCH_MODELS = {"gb_classifier" : gb_methods.gb_classifier_parameter_selection,
#                            "logistic" : log_reg_methods.logistic_regression_parameter_selection,
#                            "gb_regression" : gb_methods.gb_regression_parameter_selection}
# PARAMETER_SEARCH_OPTIONS = {"gb_classifier" : gb_methods.GB_CLASSIFIER_PARAMETERS,
#                             "logistic" : log_reg_methods.LOGISTIC_REGRESSION_PARAMETERS,
#                             "gb_regression" : gb_methods.GB_REGRESSION_PARAMETERS}

# PARAMETER_SEARCH_SCORE_THRESHOLDS = [1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5]
PARAMETER_SEARCH_SCORE_THRESHOLDS = [1e-2]  # , 1e-3, 1e-2, 0.1, 0.3, 0.5]
# PARAMETER_SEARCH_SCORE_THRESHOLDS = [1e-1] #, 1e-3, 1e-2, 0.1, 0.3, 0.5]
