import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn import linear_model
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

p = "/home/riedlinger/MetaDetect-TestEvaluation"

metrics_df = pd.read_csv(f"{p}/metrics_cat.csv")
grads_df = pd.read_csv(f"{p}/gradient_metrics.csv")
iou_df = pd.read_csv(f"{p}/true_iou.csv")

score_df = metrics_df["s"]
pure_iou_df = iou_df["true_iou"]
grads_df = grads_df.drop(["Unnamed: 0", "file_path", "dataset_box_id"], axis=1)

metrics_identifiers = grads_df.columns
norm_identifiers = [s for s in metrics_identifiers if "2-norm" in s]
norms_df = grads_df[norm_identifiers]

norms_combined_df = pd.concat([score_df, norms_df], axis=1)
grads_combined_df = pd.concat([score_df, grads_df], axis=1)

# 7
split_state = 53
x_train_score, x_test_score, y_train_score, y_test_score = sklearn.model_selection.train_test_split(
    score_df, pure_iou_df, test_size=0.5, random_state=split_state)
x_train_grads, x_test_grads, y_train_grads, y_test_grads = sklearn.model_selection.train_test_split(
    grads_df, pure_iou_df, test_size=0.5, random_state=split_state)
x_train_norms, x_test_norms, y_train_norms, y_test_norms = sklearn.model_selection.train_test_split(
    norms_df, pure_iou_df, test_size=0.5, random_state=split_state)
x_train_comb_norms, x_test_comb_norms, y_train_comb_norms, y_test_comb_norms = sklearn.model_selection.train_test_split(
    norms_combined_df, pure_iou_df, test_size=0.5, random_state=split_state)
x_train_comb_grads, x_test_comb_grads, y_train_comb_grads, y_test_comb_grads = sklearn.model_selection.train_test_split(
    grads_combined_df, pure_iou_df, test_size=0.5, random_state=split_state)

for s in [y_test_score, y_test_norms, y_test_grads, y_test_comb_norms, y_test_comb_grads]:
    print(f"GT: {len(s.values)}")
    print(f"POS: {np.sum(s.values >= 0.45)}, NEG: {np.sum(s.values < 0.45)}")

score_model = xgb.XGBClassifier(tree_method="gpu_hist", gpu_id=0,
                                max_depth=8, n_estimators=15, reg_alpha=0.5, reg_lambda=0.0)
grads_model = xgb.XGBClassifier(tree_method="gpu_hist", gpu_id=0,
                                max_depth=19, n_estimators=45, reg_alpha=0.5, reg_lambda=0.0)
norms_model = xgb.XGBClassifier(tree_method="gpu_hist", gpu_id=0,
                                max_depth=11, n_estimators=45, reg_alpha=1.5, reg_lambda=0.0)
norms_comb_model = xgb.XGBClassifier(
    tree_method="gpu_hist", gpu_id=0, max_depth=11, n_estimators=45, reg_alpha=1.5, reg_lambda=0.0)
grads_comb_model = xgb.XGBClassifier(
    tree_method="gpu_hist", gpu_id=0, max_depth=21, n_estimators=49, reg_alpha=1.5, reg_lambda=0.0)

score_baseline_model = linear_model.LogisticRegression(
    C=0.5, max_iter=5000, penalty="l2", solver="saga")

x_train_score_arr = np.array(x_train_score)[:, None]
x_test_score_arr = np.array(x_test_score)[:, None]
x_train_grads_arr = np.array(x_train_grads)
x_test_grads_arr = np.array(x_test_grads)
x_train_norms_arr = np.array(x_train_norms)
x_test_norms_arr = np.array(x_test_norms)

x_train_nc_arr = np.array(x_train_comb_norms)
x_test_nc_arr = np.array(x_test_comb_norms)
x_train_gc_arr = np.array(x_train_comb_grads)
x_test_gc_arr = np.array(x_test_comb_grads)

print("Fitting models...")
score_model.fit(x_train_score_arr, y_train_score >= 0.45)
score_baseline_model.fit(x_train_score_arr, y_train_score >= 0.45)
grads_model.fit(x_train_grads_arr, y_train_grads >= 0.45)
norms_model.fit(x_train_norms_arr, y_train_norms >= 0.45)

norms_comb_model.fit(x_train_nc_arr, y_train_comb_norms >= 0.45)
grads_comb_model.fit(x_train_gc_arr, y_train_comb_grads >= 0.45)

print("Predicting probabilities...")
y_pred_score = score_model.predict_proba(x_test_score_arr)[:, 1]
y_pred_score_baseline = score_baseline_model.predict_proba(x_test_score_arr)[
    :, 1]
y_pred_grads = grads_model.predict_proba(x_test_grads_arr)[:, 1]
y_pred_norms = norms_model.predict_proba(x_test_norms_arr)[:, 1]

y_pred_nc = norms_comb_model.predict_proba(x_test_nc_arr)[:, 1]
y_pred_gc = grads_comb_model.predict_proba(x_test_gc_arr)[:, 1]

baseline_data_df = pd.DataFrame(data=y_pred_score_baseline, columns=["y_pred"])
baseline_data_df["s"] = x_test_score_arr
baseline_data_df["true_iou"] = y_test_score.values

score_data_df = pd.DataFrame(data=y_pred_score, columns=["y_pred"])
score_data_df["s"] = x_test_score_arr
score_data_df["true_iou"] = y_test_score.values

grads_data_df = pd.DataFrame(data=y_pred_grads, columns=["y_pred"])
# grads_data_df["s"] = x_test_grads_arr
grads_data_df["true_iou"] = y_test_grads.values

norms_data_df = pd.DataFrame(data=y_pred_norms, columns=["y_pred"])
norms_data_df["true_iou"] = y_test_norms.values

dec_threshs = np.arange(0, 1, 0.01)
score_baseline_err_df = pd.DataFrame(index=dec_threshs, columns=["fn", "fp"])
score_err_df = pd.DataFrame(index=dec_threshs, columns=["fn", "fp"])
grads_err_df = pd.DataFrame(index=dec_threshs, columns=["fn", "fp"])
norms_err_df = pd.DataFrame(index=dec_threshs, columns=["fn", "fp"])

nc_err_df = pd.DataFrame(index=dec_threshs, columns=["fn", "fp"])
gc_err_df = pd.DataFrame(index=dec_threshs, columns=["fn", "fp"])

for eps_dec in dec_threshs:
    baseline_pos = y_pred_score_baseline >= eps_dec
    baseline_neg = y_pred_score_baseline < eps_dec
    score_baseline_err_df.loc[eps_dec, "fp"] = np.sum(
        np.logical_and(baseline_pos, y_test_score.values < 0.45))
    score_baseline_err_df.loc[eps_dec, "fn"] = np.sum(
        np.logical_and(baseline_neg, y_test_score.values >= 0.45))

    score_pos = y_pred_score >= eps_dec
    score_neg = y_pred_score < eps_dec
    score_err_df.loc[eps_dec, "fp"] = np.sum(
        np.logical_and(score_pos, y_test_score.values < 0.45))
    score_err_df.loc[eps_dec, "fn"] = np.sum(
        np.logical_and(score_neg, y_test_score.values >= 0.45))

    grad_pos = y_pred_grads >= eps_dec
    grad_neg = y_pred_grads < eps_dec
    grads_err_df.loc[eps_dec, "fp"] = np.sum(
        np.logical_and(grad_pos, y_test_grads.values < 0.45))
    grads_err_df.loc[eps_dec, "fn"] = np.sum(
        np.logical_and(grad_neg, y_test_grads.values >= 0.45))

    norms_pos = y_pred_norms >= eps_dec
    norms_neg = y_pred_norms < eps_dec
    norms_err_df.loc[eps_dec, "fp"] = np.sum(
        np.logical_and(norms_pos, y_test_norms.values < 0.45))
    norms_err_df.loc[eps_dec, "fn"] = np.sum(
        np.logical_and(norms_neg, y_test_norms.values >= 0.45))

    nc_pos = y_pred_nc >= eps_dec
    nc_neg = y_pred_nc < eps_dec
    nc_err_df.loc[eps_dec, "fp"] = np.sum(
        np.logical_and(nc_pos, y_test_comb_norms.values < 0.45))
    nc_err_df.loc[eps_dec, "fn"] = np.sum(
        np.logical_and(nc_neg, y_test_comb_norms.values >= 0.45))

    gc_pos = y_pred_gc >= eps_dec
    gc_neg = y_pred_gc < eps_dec
    gc_err_df.loc[eps_dec, "fp"] = np.sum(
        np.logical_and(gc_pos, y_test_comb_grads.values < 0.45))
    gc_err_df.loc[eps_dec, "fn"] = np.sum(
        np.logical_and(gc_neg, y_test_comb_grads.values >= 0.45))

base_path = "/home/riedlinger/MetaDetect-TestEvaluation/pareto_evaluation"
score_baseline_err_df.to_csv(f"{base_path}/baseline_errors.csv")
score_err_df.to_csv(f"{base_path}/score_errors.csv")
grads_err_df.to_csv(f"{base_path}/grads_errors.csv")
norms_err_df.to_csv(f"{base_path}/norms_errors.csv")

nc_err_df.to_csv(f"{base_path}/nc_errors.csv")
gc_err_df.to_csv(f"{base_path}/gc_errors.csv")


plt.plot(score_baseline_err_df["fp"], score_baseline_err_df["fn"], label="Threshold Baseline ($AuROC = {:.3}$)".format(
    sklearn.metrics.roc_auc_score(y_test_score >= 0.45, y_pred_score_baseline)))
plt.plot(score_err_df["fp"], score_err_df["fn"], label="GB Score ($AuROC = {:.3}$)".format(
    sklearn.metrics.roc_auc_score(y_test_score >= 0.45, y_pred_score)))
plt.plot(norms_err_df["fp"], norms_err_df["fn"], label="GB $|\\!|\\nabla|\\!|_2$ ($AuROC = {:.3}$)".format(
    sklearn.metrics.roc_auc_score(y_test_norms >= 0.45, y_pred_norms)))
plt.plot(nc_err_df["fp"], nc_err_df["fn"], label="GB $s + |\\!|\\nabla|\\!|_2$ ($AuROC = {:.3}$)".format(
    sklearn.metrics.roc_auc_score(y_test_comb_norms >= 0.45, y_pred_nc)))
plt.plot(grads_err_df["fp"], grads_err_df["fn"], label="GB $\\mu(\\nabla)$ ($AuROC = {:.3}$)".format(
    sklearn.metrics.roc_auc_score(y_test_grads >= 0.45, y_pred_grads)))
plt.plot(gc_err_df["fp"], gc_err_df["fn"], label="GB $s + \\mu(\\nabla)$ ($AuROC = {:.3}$)".format(
    sklearn.metrics.roc_auc_score(y_test_comb_grads >= 0.45, y_pred_gc)), c="c")
plt.xlabel("FP"), plt.ylabel("FN")
plt.legend()

plt.grid(True)

plt.savefig(f"{base_path}/pareto_fronts.pdf")
plt.savefig(f"{base_path}/pareto_fronts.png")
