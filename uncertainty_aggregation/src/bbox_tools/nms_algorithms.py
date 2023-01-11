import numpy as np
import pandas as pd
from src.bbox_tools.iou_metrics import bbox_iou


def old_nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bbox_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def nms_temp(box, bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param box/bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    best_bboxes = []

    cls_mask = (bboxes[:, 5].astype(int) == int(box[0, 5]))
    cls_bboxes = bboxes[cls_mask, :]

    best_bbox = np.asarray(box).flatten()
    best_bboxes.append(best_bbox)
    iou = bbox_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
    weight = np.zeros((len(iou),), dtype=np.float32)

    assert method in ['nms', 'soft-nms']

    if method == 'nms':
        iou_mask = iou > iou_threshold
        weight[iou_mask] = 1.0

    if method == 'soft-nms':
        weight = np.exp(-(1.0 * iou ** 2 / sigma))

    cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
    score_mask = (weight == 1.)
    cls_bboxes = cls_bboxes[score_mask]
    iou = iou[score_mask]

    iou[iou == 1] = 0

    cls_bboxes = np.concatenate(
        (np.asmatrix(cls_bboxes), np.transpose(np.asmatrix(iou))), axis=1)

    return cls_bboxes


def perform_nms_on_dataframe(df):
    img_paths = list(set(df["file_path"]))
    post_nms_df_list = []
    for p in img_paths:
        img_df = df[df["file_path"].isin([p])].drop("gradient_metrics", axis=1)
        cols = list(img_df.loc[:, "xmin":].columns)
        img_post_nms = old_nms(np.array(img_df.loc[:, "xmin":]), 0.45)
        post_nms_frame = pd.DataFrame(data=img_post_nms, columns=cols)
        post_nms_df_list.append(post_nms_frame)

    pnms_df = pd.concat(post_nms_df_list, axis=0,
                        ignore_index=True).sort_values(by="dataset_box_id")
    pnms_df.index = range(len(pnms_df))

    return pnms_df


def select_rows_from_other(df, ids_df):
    id_list = list(ids_df["dataset_box_id"])
    df = df[df["dataset_box_id"].isin(id_list)]

    return df.sort_values(by="dataset_box_id")
