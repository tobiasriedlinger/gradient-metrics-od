import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

metric = "two_norms"

result_path = f"/home/riedlinger/OD/yolov3-torch-lfs/uncertainty_metrics/kitti/retrain/post_nms_regression/{metric}_uncertainty.csv"
# result_path = "/home/riedlinger/UQ/MetaDetect-TestEvaluation/test_preparation/fusion_metrics/post_nms_regression/fusion_prediction.csv"
# result_path = "/home/riedlinger/UQ/MetaDetect-TestEvaluation/test_preparation/fusion_metrics/post_nms_regression/md_prediction.csv"
iou_path = "/home/riedlinger/OD/yolov3-torch-lfs/uncertainty_metrics/kitti/retrain/true_iou_post_nms.csv"

res_df = pd.read_csv(result_path)
iou_df = pd.read_csv(iou_path)


auroc = roc_auc_score(iou_df["true_iou"].round(0), res_df["end_score"])
r2 = r2_score(iou_df["true_iou"], res_df["end_score"])


plt.plot(iou_df["true_iou"], res_df["end_score"], "o", alpha=0.02)

plt.title("2-Norms: $AuROC = {:.3}$, $R^2 = {:.3}$".format(auroc, r2))
plt.xlabel("True $\\mathtt{IoU}$")
plt.ylabel("Predicted $\\mathtt{IoU}$")
plt.xlim(-0.05, 1.02)
plt.ylim(-0.05, 1)

plt.savefig(f"/home/riedlinger/OD/yolov3-torch-lfs/uncertainty_metrics/kitti/retrain/post_nms_regression/{metric}_plot.pdf")
plt.savefig(f"/home/riedlinger/OD/yolov3-torch-lfs/uncertainty_metrics/kitti/retrain/post_nms_regression/{metric}_plot.png")
plt.show()
