import src.models.gradient_boosting as gb_methods
import src.models.logistic_regression as log_reg_methods

# df_path = "/home/riedlinger/MetaDetect-TestEvaluation/score-baseline"
# df_path = "/home/riedlinger/MetaDetect-TestEvaluation/output-baseline"
df_path = "/home/riedlinger/MetaDetect-TestEvaluation/output+gradients"
# df_path = "/home/riedlinger/MetaDetect-TestEvaluation/non_weighted"
# df_path = "/home/riedlinger/MetaDetect-TestEvaluation/mean_weighted"

gt_path = "/home/riedlinger/datasets_ground_truth/KITTI/csv"

num_classes = 8

# TODO: this is ugly and does not generalize well...
# STANDARDIZABLE_METRICS = ["s", "prob_sum"] + [f"prob_{i}" for i in range(num_classes)] + [f"{contribution}_{weight}_{scale}_{norm}_norm" for contribution in ["cls", "conf", "loc"] for weight in ["bridge", "pred"] for scale in ["l", "m", "s"] for norm in [1, 2]]
STANDARDIZABLE_METRICS = ["s", "prob_sum"] + [f"prob_{i}" for i in range(num_classes)] + [f"{contribution}_{weight}_{norm}_norm" for contribution in [
    "cls", "conf", "loc"] for weight in ["bridge", "pred"] for norm in [1, 2]]

PARAMETER_SEARCH_MODELS = {"gb_classifier": gb_methods.gb_classifier_parameter_search_model,
                           "logistic": log_reg_methods.logistic_regression_parameter_selection}
PARAMETER_SEARCH_OPTIONS = {
    "gb_classifier": gb_methods.GB_CLASSIFIER_PARAMETERS}
