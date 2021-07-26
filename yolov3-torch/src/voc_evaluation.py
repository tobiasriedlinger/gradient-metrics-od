from src.evaluation_experiment import eval_ex

detector_settings = dict(num_classes=20,
                             weight_path="/home/OD/yolov3-torch-lfs/weights/voc2012/retrain/best_map_0.666_mf1_0.675.pt",
                             from_ckpt=True,
                             # last_n_layers="tail",
                             # reset_weights=False,
                             # unfreeze_all=True,
                             # reset_head=False
                             )

data_settings = dict(test_set="voc",
                     # train_dir="/home/datasets/COCO/2017/train2017",
                     # train_annot="/home/datasets/COCO/2017/annotations/instances_train2017.json",
                     # train_bs=32,
                     # n_cpu=8,
                     img_size=608,
                     # augmentation=True,
                     eval_dir="/home/datasets/PASCAL_VOC/test/VOCdevkit/VOC2007/JPEGImages",
                     score_thr=0.0001,
                     gt_path="/home/dataset_ground_truth/VOC2007/csv",
                     target_dir=None
                     )

eval_ex.run(config_updates=dict(detector_settings=detector_settings,
                                data_settings=data_settings))
