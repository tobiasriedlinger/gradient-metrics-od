from src.training_experiment import train_ex

detector_settings = dict(num_classes=80,
                         weight_path="/home/OD/yolov3-torch-lfs/weights/coco/retrain/ckpt_lr_1e-4e+00_ep_2_map_0.548_mf1_0.568.pt",
                         from_ckpt=True,
                         last_n_layers="tail",
                         reset_weights=False,
                         unfreeze_all=True,
                         reset_head=False
                         )

data_settings = dict(train_set="coco",
                     train_dir="/home/datasets/COCO/2017/train2017",
                     train_annot="/home/datasets/COCO/2017/annotations/instances_train2017.json",
                     train_bs=20,
                     n_cpu=8,
                     img_size=608,
                     augmentation=True,
                     eval_dir="/home/datasets/COCO/2017/val2017",
                     score_thr=0.01,
                     gt_path="/home/dataset_ground_truth/COCO/2017/csv"
                     )

train_settings = dict(cpu_only=False,
                      ckpt_dir="/home/OD/yolov3-torch-lfs/weights/coco/retrain",
                      save_epochs=1,
                      phase_epochs=3,
                      initial_lr=1E-4,
                      num_phases=2,
                      log_dir="../log",
                      verbose=False,
                      debug=False
                      )

r = train_ex.run(config_updates=dict(train_settings=train_settings,
                                     detector_settings=detector_settings,
                                     data_settings=data_settings))
