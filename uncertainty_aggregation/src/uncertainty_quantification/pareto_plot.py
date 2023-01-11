import pandas as pd
import matplotlib.pyplot as plt

POS = 0
NEG = 0

p = "/home/riedlinger/MetaDetect-TestEvaluation/grads_lr=1e-3_bs=64/pareto_evaluation"

bl_errors = pd.read_csv(f"{p}/baseline_errors.csv")
score_errors = pd.read_csv(f"{p}/score_errors.csv")
norms_errors = pd.read_csv(f"{p}/norms_errors.csv")
grads_errors = pd.read_csv(f"{p}/grads_errors.csv")
nc_errors = pd.read_csv(f"{p}/nc_errors.csv")
gc_errors = pd.read_csv(f"{p}/gc_errors.csv")

auc_scores = {}

labels = ["Threshold Baseline", "GB Score", "GB $|\!|\\nabla|\!|$",
          "GB $\\mu(\\nabla)$", "GB $s + |\!|\\nabla|\!|$", "GB $s + \\mu(\\nabla)$"]

for i, df in enumerate([bl_errors, score_errors, norms_errors, grads_errors, nc_errors, gc_errors]):
    df["tpr"] = 1 - df["fn"]/POS
    df["fpr"] = df["fp"]/NEG
