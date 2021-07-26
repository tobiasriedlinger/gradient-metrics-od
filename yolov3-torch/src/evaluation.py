import numpy as np
import pandas as pd
import time
import glob


def make_predictions(df):
    """
    Perform image-wise NMS on df, resulting in prediction.
    Args:
        df (pd.DataFrame): DataFrame containing the pre-NMS prediction of an object detector. Columns must contain
                ["file_path", "xmin", "ymin", "xmax", "ymax", "s", "category_idx"] (order of the last 6 is important).
    Returns:
        df_pred (pd.DataFrame): DataFrame containing the post-NMS bounding boxes (NMS survivors).
    """

    all_image_paths = list(set(df["file_path"]))
    df_pred = None
    for i in range(len(all_image_paths)):

        box_mask = (df.loc[:, 'file_path'] == all_image_paths[i]).T

        bboxes_single_image = df.loc[box_mask, 'xmin':'category_idx']
        bboxes_single_image = np.asarray(bboxes_single_image.to_numpy()).astype(float)

        bboxes_single_image = old_nms(bboxes_single_image, 0.45)
        bboxes_single_image = np.asmatrix(bboxes_single_image)
        df_single_image = pd.DataFrame(data=bboxes_single_image)
        df_single_image['6'] = all_image_paths[i]

        if i == 0:
            df_pred = df_single_image
        else:
            df_pred = df_pred.append(df_single_image, ignore_index=True)

    df_pred.columns = ['xmin', 'ymin', 'xmax', 'ymax', 's', 'category_idx', 'file_path']
    cols = ['file_path', 'xmin', 'ymin', 'xmax', 'ymax', 's', 'category_idx']
    df_pred = df_pred[cols]

    return df_pred


def calculate_map(df,
                  gt_path,
                  print_results=True,
                  num_classes=8):
    """
    Computes mean Average Precision (mAP) and mean F1-Score over predictions in df on the dataset whose ground truth is
    image-wise saved in gt_path.
    Args:
        df (pd.DataFrame):  DataFrame containing pre-NMS detections with formatting like make_prediction.
        gt_path (str): Path to directory containing image-wise ground truth data in .csv format.
        print_results (bool): Flag indicating whether to output computed partial results such as class-wise AP.
        num_classes (int): Number of classes in the dataset (needed for loop over classes).
    Returns:
        mean_ap (list[float]): List of computed class-wise AP scores.
        f1 (list[float]): List of computed class-wise F1 scores.
    """
    np.seterr(divide='ignore', invalid='ignore')

    df = df.loc[:, 'file_path':'category_idx']
    df["s"] = pd.to_numeric(df["s"], downcast="float")
    df = df.reset_index(drop=True)

    gt = get_gt(gt_path)
    gt['used'] = 0
    df = make_predictions(df)

    df["s"] = pd.to_numeric(df["s"], downcast="float")

    df["category_idx"] = pd.to_numeric(df["category_idx"], downcast="integer")
    df = df.sort_values(by=['s'], ascending=False)

    mean_ap = np.zeros(num_classes)
    f1_scores = []
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
                boxes_gt = np.array(gt_class_single_image.loc[:, "xmin":"ymax"])
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

        f1_rec = recall[-1]
        f1_prec = precision[-1]
        f1_scores.append(2 * f1_rec * f1_prec / (f1_rec + f1_prec))

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

        if print_results:
            print(ap)
        mean_ap[c] = ap

    f1 = np.array(f1_scores)
    if print_results:
        print(mean_ap)
        print(f"mAP : {np.nanmean(mean_ap)}")
        print(f1)
        print(f"mF1 : {np.nanmean(f1)}")
    return mean_ap, f1


def old_nms(bboxes,
            iou_threshold,
            sigma=0.3,
            method='nms'):
    """

    Args:
        bboxes (np.Array): Array of dimension [N, 6] where formatting of the latter is [xmin, ymin, xmax, ymax, s, c]
        iou_threshold (float): Threshold between 0 and 1 for cluster recognition.
        sigma (float): Optional smoothing strength in case of soft NMS.
        method (str): Either "nms" or "soft-nms", indicating NMS algorithm to be used.

    Returns:

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
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
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


def bbox_iou(boxes1, boxes2):
    """
    IoU routine used in old_nms.
    Args:
        boxes1 (np.Array): First array of boxes for IoU computation.
        boxes2 (np.Array): Second array of boxes for IoU computation.

    Returns:
        ious (np.Array): Array of computed inter-list IoUs.
    """

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = 1.0 * inter_area / union_area

    return ious


def get_gt(gt_path):
    """
    Collects ground truth boxes from given .csv-directory.
    Args:
        gt_path (str): .csv-path to folder containing ground truth data.
    Returns:
        gt (pd.DataFrame): Concatenated ground truth DataFrames.
    """
    gt = None
    for i, p in enumerate(glob.glob(str(gt_path)+"/*.csv")):
        gt_df = load_single_frame(p, verbose=False)
        if i == 0:
            gt = gt_df
        else:
            gt = gt.append(gt_df, ignore_index=True)

    return gt


def load_single_frame(path, verbose=False):
    """
    Cleanly loads .csv file (deletes re-named index "Unnamed: 0" from table).
    Args:
        path (str): Path to .csv-file to be read.
        verbose (bool): Flag indicating whether to comment reading time.
    Returns:
        df (pd.DataFrame): Contents of .csv-file encoded in a DataFrame.
    """
    time_0 = time.time()
    if verbose:
        print("Loading dataframe...")
    try:
        df = pd.read_csv(path).drop("Unnamed: 0", axis=1)
    except KeyError:
        df = pd.read_csv(path)
    if verbose:
        print("Done. ({:.3}s)".format(time.time() - time_0))

    return df
