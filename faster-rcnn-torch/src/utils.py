import torch


def xywh_to_cxcywh(bbox):
    bbox[..., 0] += bbox[..., 2] / 2
    bbox[..., 1] += bbox[..., 3] / 2
    return bbox


def xywh_to_border(bbox):
    bbox[..., 2] += bbox[..., 0]
    bbox[..., 3] += bbox[..., 1]
    return bbox


def cxcywh_to_border(coordinates):
    xmin = coordinates[..., 0] - coordinates[..., 2] / 2
    ymin = coordinates[..., 1] - coordinates[..., 3] / 2
    xmax = coordinates[..., 0] + coordinates[..., 2] / 2
    ymax = coordinates[..., 1] + coordinates[..., 3] / 2

    return xmin, ymin, xmax, ymax


def untransform_bboxes(bboxes, scale, padding):
    """transform the bounding box from the scaled image back to the unscaled image."""
    # xmin = bboxes[..., 0]
    # ymin = bboxes[..., 1]
    # xmax = bboxes[..., 2]
    # ymax = bboxes[..., 3]
    x = 0.5 * (bboxes[..., 0] + bboxes[..., 2])
    y = 0.5 * (bboxes[..., 1] + bboxes[..., 3])
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    # x, y, w, h = bbs
    # xmin /= scale
    # ymin /= scale
    # xmax /= scale
    # ymax /= scale
    # xmin -= padding[0]
    # xmax -= padding[0]
    # ymin -= padding[1]
    # ymax -= padding[1]
    x /= scale
    y /= scale
    w /= scale
    h /= scale
    x -= padding[0]
    y -= padding[1]
    return torch.stack([x - w/2, y - h/2, x + w/2, y + h/2], dim=1)
