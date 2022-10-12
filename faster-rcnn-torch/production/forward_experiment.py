"""
Routine for network forward pass, e.g. for evaluation or MC Dropout inference.
"""

from sacred import Experiment

from production.get_forwards import ForwardConfig, GetForwards

forward_ex = Experiment("Forward Pass Experiment")


@forward_ex.config
def forward_config():
    detector_settings = dict(num_classes=80,
                             weight_path="/home/riedlinger/OD/faster-rcnn-torch-lfs/weights/resnet50-19c8e357.pth",
                             weight_is_ckpt=False,
                             dropout_rate=0.0,
                             score_thresh=1e-4
                             )

    data_settings = dict(img_dir="/home/datasets/COCO/2017/val2017",
                         batch_size=8,
                         img_size=800,
                         target_folder="/home/riedlinger/OD/faster-rcnn-torch-lfs/detection_eval/coco2017"
                         )


@forward_ex.automain
def forward_main(detector_settings, data_settings):
    det_s = detector_settings
    data_s = data_settings
    cfg = ForwardConfig()
    cfg.weight_path = det_s["weight_path"]
    cfg.weight_is_ckpt = det_s["weight_is_ckpt"]
    cfg.img_size = data_s["img_size"]
    cfg.img_dir = data_s["img_dir"]
    cfg.batch_size = data_s["batch_size"]
    cfg.target_folder = data_s["target_folder"]
    cfg.score_thresh = det_s["score_thresh"]
    cfg.num_classes = det_s["num_classes"]
    cfg.dropout_rate = det_s["dropout_rate"]

    GetForwards(config=cfg).run_forward()
