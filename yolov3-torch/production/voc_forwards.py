from production.forward_pass_experiment import forward_ex
detector_settings = dict(num_classes=20,
                         weight_path="/home/OD/yolov3-torch-lfs/weights/voc2012/retrain/ckpt_lr_1e-5e+00_ep_29_map_0.642_mf1_0.709.pt",
                         weight_is_ckpt=True,
                         dropout_rate=0.0,
                         score_thresh=1e-2
                         )

data_settings = dict(img_dir="/home/datasets/PASCAL_VOC/test/VOCdevkit/VOC2007/JPEGImages",
                     batch_size=8,
                     img_size=608,
                     target_folder="/home/OD/yolov3-torch-lfs/detection_eval/voc2012/retrain/csv"
                     )

forward_ex.run(config_updates=dict(detector_settings=detector_settings,
                                   data_settings=data_settings))
