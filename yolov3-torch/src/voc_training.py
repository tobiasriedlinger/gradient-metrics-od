from src.training_experiment import train_ex

detector_settings = dict(num_classes=20,
                         # weight_path="/home/OD/yolov3-torch-lfs/weights/"
                         # + "yolov3_original.pt",
                         weight_path="/home/OD/yolov3-torch-lfs/weights/voc2012/retrain/"
                                     + "ckpt_lr_1e-5e+00_ep_69_map_0.697_mf1_0.634.pt",
                         from_ckpt=True,
                         last_n_layers="tail",
                         reset_weights=False,
                         unfreeze_all=True,
                         reset_head=False
                         )

data_settings = dict(train_set="voc",
                     train_dir="/home/datasets/PASCAL_VOC/train",
                     train_annot="2012",
                     train_bs=22,
                     n_cpu=8,
                     img_size=608,
                     augmentation=True,
                     eval_dir="/home/datasets/PASCAL_VOC/test/VOCdevkit/VOC2007/JPEGImages",
                     score_thr=0.01,
                     gt_path="/home/dataset_ground_truth/VOC2007/csv"
                     )

train_settings = dict(cpu_only=False,
                      ckpt_dir="/home/OD/yolov3-torch-lfs/weights/voc2012/retrain",
                      save_epochs=5,
                      phase_epochs=30,
                      initial_lr=1E-5,
                      num_phases=2,
                      log_dir="../log",
                      verbose=False,
                      debug=False
                      )

r = train_ex.run(config_updates=dict(train_settings=train_settings,
                                     detector_settings=detector_settings,
                                     data_settings=data_settings))
