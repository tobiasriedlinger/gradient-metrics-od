import pandas as pd
import numpy as np

import configs.data_config as data_cfg
from src.bbox_tools.nms_algorithms import old_nms
from src.bbox_tools.iou_metrics import bbox_iou
from src.uncertainty_quantification.true_iou import get_true_iou
from src.api.loading_data import load_single_frame, get_gt


def make_predictions(df):

    all_image_paths = list(set(df["file_path"]))
    for i in range(len(all_image_paths)):

        box_mask = (df.loc[:, 'file_path'] == all_image_paths[i]).T

        bboxes_single_image = df.loc[box_mask, 'xmin':'category_idx']
        bboxes_single_image = np.asarray(
            bboxes_single_image.to_numpy()).astype(float)

        bboxes_single_image = old_nms(bboxes_single_image, 0.45)
        bboxes_single_image = np.asmatrix(bboxes_single_image)
        df_single_image = pd.DataFrame(data=bboxes_single_image)
        df_single_image['6'] = all_image_paths[i]
        try:
            df_pred = df_pred.append(df_single_image, ignore_index=True)
        except:
            df_pred = df_single_image

    df_pred.columns = ['xmin', 'ymin', 'xmax',
                       'ymax', 's', 'category_idx', 'file_path']
    cols = ['file_path', 'xmin', 'ymin', 'xmax', 'ymax', 's', 'category_idx']
    df_pred = df_pred[cols]

    return df_pred


def calculate_map(df, gt_path):

    df = df.loc[:, 'file_path':'category_idx']
    df["s"] = pd.to_numeric(df["s"], downcast="float")
    df = df.reset_index(drop=True)

    gt = get_gt(gt_path)
    gt['used'] = 0
    df = make_predictions(df)
    iou_df = get_true_iou(df, data_cfg.default_gt_path, keep_ids=False)
    df.to_csv(f"{data_cfg.default_df_path}/regression_test/metrics.csv")
    iou_df.to_csv(f"{data_cfg.default_df_path}/regression_test/true_iou.csv")

    df["s"] = pd.to_numeric(df["s"], downcast="float")

    df["category_idx"] = pd.to_numeric(df["category_idx"], downcast="integer")
    df = df.sort_values(by=['s'], ascending=False)

    num_classes = 8
    mean_ap = np.zeros(num_classes)
    for c in range(num_classes):
        df_class = df.loc[df['category_idx'] == c]
        tp = np.zeros(df_class.shape[0] + 1)
        fp = np.zeros(df_class.shape[0] + 1)
        gt_class = gt.loc[gt['category_idx'] == c]
        index = df_class.index.tolist()
        for i in range(len(index)):
            image_path = df_class.loc[index[i], 'file_path']
            gt_class_single_image = gt_class.loc[gt_class['file_path'] == image_path]
            index_gt = gt_class_single_image.index.tolist()
            if gt_class_single_image.shape[0] > 0:
                boxes_pred = np.array(df_class.loc[index[i], "xmin":"ymax"])
                boxes_gt = np.array(
                    gt_class_single_image.loc[:, "xmin":"ymax"])
                iou = bbox_iou(boxes_pred, boxes_gt)
                if np.max(iou) >= 0.5:
                    if gt.loc[index_gt[np.argmax(iou)], 'used'] == 0:
                        gt.loc[index_gt[np.argmax(iou)], 'used'] = 1
                        tp[i + 1] = tp[i] + 1
                        fp[i + 1] = fp[i]
                    else:
                        tp[i + 1] = tp[i]
                        fp[i + 1] = fp[i] + 1
                else:
                    tp[i + 1] = tp[i]
                    fp[i + 1] = fp[i] + 1
            else:
                tp[i + 1] = tp[i]
                fp[i + 1] = fp[i] + 1

        summe_gt = gt_class.shape[0]
        recall = tp / summe_gt
        precision = tp / (tp + fp)
        recall = np.asarray(recall)
        precision = np.asarray(precision)

        recall = np.append(recall, 1.0)
        mrec = recall[:]
        precision = np.append(precision, 0.0)
        mpre = precision[:]

        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)

        ap = 0.0
        for i in i_list:
            ap += ((mrec[i] - mrec[i - 1]) * mpre[i])

        print(ap)
        mean_ap[c] = ap

    print(mean_ap)
    print(np.nanmean(mean_ap))


if __name__ == '__main__':
    gt_path = "/home/riedlinger/dataset_ground_truth/KITTI/csv"
    df = load_single_frame(
        '/home/riedlinger/MetaDetect-TestEvaluation/test_marius/metrics_cat.csv')
    calculate_map(df, gt_path)
