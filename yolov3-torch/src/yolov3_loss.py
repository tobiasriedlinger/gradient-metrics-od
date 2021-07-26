"""
YOLOv3 loss funcitonalities
"""
import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce, mse_loss as mse
import numpy as np

from src.config import EPSILON, ANCHORS
from src.training_old import iou_batch

# Threshold for IoU between anchor boxes and target boxes for negative assignment (see below)
neg_iou_thresh = 0.5
# Same for positive assignment.
pos_iou_thresh = 0.5


def yolov3_loss(detection, tgt, tgt_len, img_size, average=True):
    """
    Main loss function used for training. CrossEntropy loss for score, probabilities and bounding box centers;
    MeanSquaredError loss for bounding box width and height.
    Args:
        detection (tuple(list(torch.Tensor))): Tuple with entries for the three scales (s, m, l), each containing a list
                with with 0: raw network output [B, Anc, G^2, Attr] and 1: transformed network output [B, N_pred, Attr]
                (w.r.t. rescaled image). B: batch size, Anc: number of anchors per scale, G: grid side length for scale,
                N_pred: total number of predictions, i.e. Anc * (1^2 + 2^2 + 4^2) * G_min^2, Attr: number of attributes
                per box, i.e. (4 + 1) + n_classes
        tgt (torch.Tensor): Tensor [B, N_max, Attr] containing ground truth boxes for current batch. N_max is the
                maximum number of gts in the batch, the rest is filled with zeros (see data loaders).
        tgt_len (torch.Tensor): Tensor [B,] containing the number of gt instances per batch entry.
        img_size (int): Input image size, needed for determining grid sizes.
        average (bool): Whether to average the loss over batch size.
    Returns:
        (tuple(torch.Tensor)): Total, localization, confidence and probability loss scalar.
    """
    n_batch, n_tgt_max, n_attributes = tgt.size()

    raw_pred = torch.cat([detection[i][0].view(n_batch, -1, n_attributes) for i in range(3)], dim=1)

    tgt_t, pos_mask, neg_mask = get_targets(tgt, tgt_len, img_size)

    loc_bce = torch.sum(bce(raw_pred[:, :, :2], tgt_t[:, :, :2], reduction="none"), dim=2)
    loc_mse = torch.sum(mse(raw_pred[:, :, 2:4], tgt_t[:, :, 2:4], reduction="none"), dim=2)
    loc_loss = 2 * torch.sum(loc_bce[pos_mask] + loc_mse[pos_mask])

    conf_bce = bce(raw_pred[:, :, 4], tgt_t[:, :, 4].float(), reduction="none")
    conf_loss = torch.sum(conf_bce[pos_mask | neg_mask])

    prob_bce = torch.sum(bce(raw_pred[:, :, 5:], tgt_t[:, :, 5:].float(), reduction="none"), dim=2)
    prob_loss = torch.sum(prob_bce[pos_mask])

    total_loss = loc_loss + conf_loss + prob_loss

    if average:
        total_loss = total_loss / n_batch

    return total_loss, loc_loss, conf_loss, prob_loss


def get_multi_level_anchors(n_batch, img_size):
    """
    Generates all possible ground truth anchor boxes given image and batch sizes.
    Args:
        n_batch (int): Batch size
        img_size (int): Image side length in pixels.
    Returns:
        (tuple(torch.Tensor)): 0: Anchors for given batch [B, N_pred, 4] and 1: Corresponding image strides [B,].
    """
    anchors, strides = [], []
    wh_anchor = torch.tensor(ANCHORS).float().cuda()
    for s in [8, 16, 32]:
        scale = int(np.log2(s / 8))
        scale_anchors = wh_anchor[3*scale:3*(scale+1), :]
        grid_size = img_size // s
        grid_tensor = torch.arange(grid_size, dtype=torch.float, device=wh_anchor.device).repeat(grid_size, 1)
        anchor_x_grid = grid_tensor.view([1, 1, grid_size, grid_size]).repeat(n_batch, 3, 1, 1).cuda()
        anchor_y_grid = grid_tensor.t().view([1, 1, grid_size, grid_size]).repeat(n_batch, 3, 1, 1).cuda()

        x_anchor = s * (anchor_x_grid + 0.5)  # [B, Anc, G, G]
        y_anchor = s * (anchor_y_grid + 0.5)
        w_anchor = scale_anchors[:, 0].view([1, 3, 1, 1]).repeat(n_batch, 1, grid_size, grid_size)
        h_anchor = scale_anchors[:, 1].view([1, 3, 1, 1]).repeat(n_batch, 1, grid_size, grid_size)
        bbox_anchor = torch.stack((x_anchor, y_anchor, w_anchor, h_anchor), dim=4).view(n_batch, -1, 4)
        n_scale_anchors = bbox_anchor.size(1)
        anchors.append(bbox_anchor)
        strides.append(torch.full([n_scale_anchors], s, dtype=torch.float).cuda())

    return torch.cat(anchors, dim=1), torch.cat(strides, dim=0)


def get_responsible_anchor_ids(tgt, tgt_len, img_size):
    """
    For a list of N_pred entries, find the indices of those anchor boxes responsible for given target boxes.
    Args:
        tgt (torch.Tensor): See yolov3_loss above [B, N_max, Attr].
        tgt_len (torch.Tensor): See yolov3_loss above [B,].
        img_size (int): Image size.
    Returns:
        (torch.Tensor): Responsibility flags for tgt [B, N_pred].
    """
    n_batch = tgt.size(0)
    resp_grids = []
    for s in [8, 16, 32]:
        grid_size = img_size // s
        grid_x = (tgt[..., 0] // s).long()
        grid_y = (tgt[..., 1] // s).long()
        grid_ids = grid_size * grid_y + grid_x
        resp_grid = torch.zeros(n_batch, 3, grid_size**2)

        for b in range(n_batch):
            for j in range(tgt_len[b]):
                resp_grid[b, :, (grid_ids[b, j]).long()] = 1
        resp_grids.append(resp_grid.view(n_batch, -1))

    return torch.cat(resp_grids, dim=1).bool().cuda()


def assign_targets(tgt, tgt_len, anchors, resp_grid):
    """
    Assignment of individual anchors to given target boxes to either background (-1), negative (0) or positive (int > 0)
    Args:
        tgt (torch.Tensor): See yolov3_loss above [B, N_max, Attr].
        tgt_len (torch.Tensor): See yolov3_loss above [B,].
        anchors (torch.Tensor): Multi-level anchors for current batch [B, N_pred, 4].
        resp_grid (torch.Tensor): Responsibility flags for current target batch [B, N_pred].
    Returns:
        assignment (torch.Tensor): Assignment result for targets to anchors.
    """
    n_batch, n_pred, _ = anchors.size()
    iou_anchors = iou_batch(anchors, tgt[..., :4], center=True)

    # Initial assignment of background class.
    assignment = torch.full([n_batch, n_pred], -1, dtype=torch.long).cuda()

    max_gt_iou, argmax_gt_iou = torch.max(iou_anchors, dim=2)
    # Negative class (possibly subject to change)
    # Anchor is negative if it has 0 <= max_{all gts} IoU <= neg_iou_thresh.
    # Boxes from negative anchors will contribute to confidence loss.
    assignment[(max_gt_iou >= 0) & (max_gt_iou <= neg_iou_thresh)] = 0

    # Set iou for non-responsible anchors to irrelevant.
    iou_anchors[~resp_grid, :] = -1.

    # Assign positives in the usual way:
    # Positive anchors have max_{all gts} IoU > pos_iou_thresh and have True responsibility flag.
    # Assignment is set to gt with maximum overlap; index shifted to prevent confusion with negative anchors.
    max_anc_iou, argmax_anc_iou = torch.max(iou_anchors, dim=1)
    max_gt_iou, argmax_gt_iou = torch.max(iou_anchors, dim=2)

    pos_ids = (max_gt_iou > pos_iou_thresh) & resp_grid
    assignment[pos_ids] = argmax_gt_iou[pos_ids] + 1

    # Alternatively, to each gt, we assign the responsible anchor with maximal IoU as positive.

    for b in range(n_batch):
        for j in range(tgt_len[b]):
            if max_anc_iou[b, j] > 0.0:
                iou_ids = (max_gt_iou[b, :] == max_anc_iou[b, j]) & resp_grid[b, :]
                assignment[b, iou_ids] = j + 1
            else:
                print("Found target with no matching prior!!!")

    return assignment


def get_targets(tgt, tgt_len, img_size):
    """
    Transforms targets to grid level for each of the three scales.
    Args:
        tgt (torch.Tensor): See yolov3_loss above [B, N_max, Attr].
        tgt_len (torch.Tensor): See yolov3_loss above [B,].
        img_size (int): Image size.
    Returns:
        (tuple(torch.Tensor)): Containing 0: Target tensor [B, N_pred, Attr], 1: Positive mask indicating where gts
                    were matched to anchors [B, N_pred], 2: Negative mask indicating where anchors had insufficient
                    overlap with any ground truth [B, N_pred].
    """
    n_batch, n_max, n_attributes = tgt.size()
    anchors, strides = get_multi_level_anchors(n_batch, img_size)
    resp_grid = get_responsible_anchor_ids(tgt, tgt_len, img_size)
    assignment = assign_targets(tgt, tgt_len, anchors, resp_grid)
    neg_mask = (assignment == 0)
    pos_mask = (assignment > 0)

    tgt_t = torch.zeros([n_batch, assignment.size(1), n_attributes], dtype=torch.float, device=tgt.device)

    for b in range(n_batch):
        for j in range(tgt_len[b]):
            pred_placement = (assignment[b, :] == j+1)
            n_assigns = torch.sum(pred_placement).cuda()
            g_x = (tgt[b, j, 0] // strides[pred_placement]).long()
            g_y = (tgt[b, j, 1] // strides[pred_placement]).long()
            t_x = (tgt[b, j, 0] / strides[pred_placement] - g_x).clamp(EPSILON, 1-EPSILON).view(-1, 1)
            t_y = (tgt[b, j, 1] / strides[pred_placement] - g_y).clamp(EPSILON, 1-EPSILON).view(-1, 1)

            t_w = torch.log((tgt[b, j, 2] / anchors[b, pred_placement, 2]).clamp(min=EPSILON)).view(-1, 1)
            t_h = torch.log((tgt[b, j, 3] / anchors[b, pred_placement, 3]).clamp(min=EPSILON)).view(-1, 1)

            tgt_t[b, pred_placement, :] = torch.cat([t_x, t_y, t_w, t_h,
                                                     tgt[b, j, 4:].view(1, -1).repeat(n_assigns, 1)],
                                                    dim=-1)
    assert torch.sum(pos_mask) == torch.sum(tgt_t[..., 4])

    return tgt_t, pos_mask, neg_mask
