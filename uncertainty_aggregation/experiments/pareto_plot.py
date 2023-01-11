import pandas as pd
import matplotlib.pyplot as plt

from sacred import Experiment

ex = Experiment("plot_experiment")

@ex.config
def plot_config():
    err_path = "/home/riedlinger/OD/yolov3-torch-lfs/uncertainty_metrics/kitti/retrain/post_nms_evaluation"

@ex.capture
def plot_curve(metric_id, err_path):
    err_df = pd.read_csv(f"{err_path}/{metric_id}_errs.csv")
    plt.plot(err_df["fp"], err_df["fn"], label=metric_id, linewidth=1)

@ex.automain
def plot_main(err_path):
    # plot_curve("two_norms")
    plot_curve("meta_detect")
    plot_curve("gradients")
    plot_curve("mc_dropout_std")
    # plot_curve("mc+meta_detect")
    # plot_curve("score_baseline_thresholding")
    plot_curve("score_baseline")

    plt.legend()
    plt.xlim(-100, 3000)
    plt.ylim(-200, 6000)

    plt.show()
