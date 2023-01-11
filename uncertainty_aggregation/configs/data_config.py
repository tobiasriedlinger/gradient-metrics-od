default_df_path = "/home/riedlinger/MetaDetect-TestEvaluation/test_preparation"

default_gt_path = "/home/riedlinger/dataset_ground_truth/KITTI/csv"

num_classes = 8

SCORE_THRESHOLDS_META_DETECT = [0.01, 0.1, 0.3, 0.5]

# # #
#       Metrics identifiers used in DataFrame
# # #
OUTPUT_METRICS = ["xmin", "ymin", "xmax", "ymax", "s", "category_idx",
                  "prob_sum"] + [f"prob_{i}" for i in range(num_classes)]
STD_OUTPUT_METRICS = ["s", "prob_sum"] + \
    [f"prob_{i}" for i in range(num_classes)]
META_DETECT_METRICS = ['Number of Candidate Boxes', 'x_min',
                       'x_max', 'x_mean', 'x_std', 'y_min', 'y_max', 'y_mean', 'y_std', 'w_min', 'w_max', 'w_mean', 'w_std',
                       'h_min', 'h_max', 'h_mean', 'h_std', 'size', 'size_min', 'size_max', 'size_mean', 'size_std',
                       'circum', 'circum_min', 'circum_max', 'circum_mean', 'circum_std', 'size/circum', 'size/circum_min',
                       'size/circum_max', 'size/circum_mean', 'size/circum_std', 'score_min', 'score_max', 'score_mean', 'score_std',
                       'IoU_pb_min', 'IoU_pb_max', 'IoU_pb_mean', 'IoU_pb_std']
