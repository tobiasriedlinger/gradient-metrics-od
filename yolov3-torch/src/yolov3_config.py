"""
Default configuration container for YOLOv3 net.
"""


class YOLOV3Config(object):
    img_size = 608
    dropout_rate = 0.5
    eval_score_thr = 0.1

    num_classes = 80
    num_attrib = num_classes + 5
    last_layer_dim = 3 * num_attrib
