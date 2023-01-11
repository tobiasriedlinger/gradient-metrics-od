import pandas as pd
import numpy as np
import tqdm
import cv2
from src.bbox_tools.nms_algorithms import old_nms

pred_path = "/home/riedlinger/MetaDetect-TestEvaluation/metrics_cat.csv"
iou_path = "/home/riedlinger/MetaDetect-TestEvaluation/true_iou.csv"

score_thresh = 0.3

pred_df = pd.read_csv(pred_path)
iou_df = pd.read_csv(iou_path)

path_list = list(set(pred_df["file_path"]))

for p in tqdm.tqdm(path_list):
    has_bad_sample = False
    img_id = p.split("/")[-1].split(".")[0]
    img_pred_df = pred_df[pred_df["file_path"].isin([p])]
    img_iou_df = iou_df[iou_df["dataset_box_id"].isin(
        list(img_pred_df["dataset_box_id"]))]

    gt_df = pd.read_csv(
        f"/home/riedlinger/dataset_ground_truth/KITTI/csv/{img_id}_gt.csv")

    base_img = cv2.imread(p)

    gt_img = base_img.copy()
    coords = ["xmin", "ymin", "xmax", "ymax"]

    for i in gt_df.index:
        b = gt_df.loc[i, coords]
        cv2.rectangle(gt_img, (b[0], b[1]), (b[2], b[3]),
                      color=(0, 165, 255), thickness=2)

    data_df = pd.concat([img_pred_df, img_iou_df], axis=1)
    data_df = data_df[data_df["s"] >= score_thresh]

    pred_data = data_df[coords+["s", "category_idx", "true_iou"]]

    pred = old_nms(np.array(pred_data), iou_threshold=0.45)
    pred_img = base_img.copy()
    for b in pred:
        cv2.rectangle(pred_img, (int(b[0]), int(b[1])), (int(
            b[2]), int(b[3])), color=(200, 255, 200), thickness=1)

    score_img = pred_img.copy()

    for b in pred:
        if b[6] < 0.1:
            c = (200, 200, 255)
            has_bad_sample = True
        else:
            c = (200, 255, 200)
        s_str = "s={:.2}/IoU={:.2}".format(b[4], b[6])
        t_size = cv2.getTextSize(s_str, 0, 0.5, thickness=1)[0]
        cv2.rectangle(score_img, (int(b[0]), int(b[1]) - t_size[1] - 1), (int(b[0]) + t_size[0], int(b[1])),
                      c, -1)
        cv2.putText(score_img, s_str, (int(b[0]), int(b[1]) - 1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(0, 0, 0), thickness=1)

    if has_bad_sample:
        coll = np.vstack([gt_img, score_img])
        cv2.imwrite(
            f"/home/riedlinger/MetaDetect-TestEvaluation/prediction_collages/pred_collage_{img_id}.png", coll)
