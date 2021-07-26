import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from sacred import Experiment

from src.access import config_device, load_yolov3_model, load_dataset, save_checkpoint_weight_file
from src.utils import init_layer_randomly
from src.yolov3_loss import yolov3_loss
from src.evaluation import calculate_map

from production.get_forwards import ForwardConfig, GetForwards

train_ex = Experiment("Train YOLOv3")


@train_ex.config
def train_cfg():
    # Detector Settings
    detector_settings = dict(num_classes=80,
                             weight_path="/home/OD/yolov3-torch-lfs/weights/yolov3_original.pt",
                             from_ckpt=True,
                             last_n_layers="tail",
                             reset_weights=False,
                             unfreeze_all=True,
                             reset_head=False
                             )

    data_settings = dict(train_set="coco",
                         train_dir="/home/datasets/COCO/2017/train2017",
                         train_annot="/home/datasets/COCO/2017/annotations/instances_train2017.json",
                         train_bs=32,
                         n_cpu=8,
                         img_size=608,
                         augmentation=True,
                         eval_dir="/home/datasets/COCO/2017/val2017",
                         score_thr=0.1,
                         gt_path="/home/dataset_ground_truth/COCO/2017/csv"
                         )

    train_settings = dict(cpu_only=False,
                          ckpt_dir="/home/OD/yolov3-torch-lfs/weights/coco/retrain",
                          save_epochs=3,
                          phase_epochs=30,
                          initial_lr=1E-4,
                          num_phases=4,
                          log_dir="../log",
                          verbose=False,
                          debug=False
                          )

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def phase(model,
          n_epochs,
          learning_rate,
          train_dl,
          det_s,
          data_s,
          train_s):

    dev = config_device(train_s["cpu_only"])
    optimizer = torch.optim.Adam(
        filter(lambda par: par.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.0005
    )
    print(f"{n_epochs} epochs with initial learning rate {learning_rate}.")
    scheduler = None

    for epoch_i in range(n_epochs):

        if epoch_i == 0:
            scheduler = warmup_lr_scheduler(optimizer, 200, 0.1)
        # else:
        #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        #                                                 step_size=2,
        #                                                 gamma=0.1
        #                                                 )
        lr = getattr(optimizer, "param_groups")[-1]["lr"]

        print(f"Epoch {epoch_i}, LR = {lr}")
        cumul_losses = torch.zeros(3, dtype=torch.float).cuda()
        for batch_i, (imgs, targets, target_lengths) in enumerate(tqdm(train_dl)):
            with torch.autograd.detect_anomaly():
                # print(getattr(optimizer, "param_groups")[-1]["lr"])
                optimizer.zero_grad()
                imgs = imgs.to(dev)
                targets = targets.to(dev)
                target_lengths = target_lengths.to(dev)
                result = model(imgs)
                try:
                    losses = yolov3_loss(result,
                                         targets,
                                         target_lengths,
                                         data_s["img_size"],
                                         True  # Average
                                         )
                    losses[0].backward()
                except RuntimeError:
                    optimizer.zero_grad()
                    continue
                optimizer.step()
                if epoch_i == 0:
                    scheduler.step()
            cumul_losses += torch.stack(losses[1:]).detach()
        print("Loc Loss: {:.5}, Conf Loss: {:.5}, Class Loss: {:.5}".format(cumul_losses[0].item(),
                                                                            cumul_losses[1].item(),
                                                                            cumul_losses[2].item()))

        if (epoch_i + 1) % train_s["save_epochs"] == 0:
            fwd_cfg = ForwardConfig()
            predictions = pd.concat(GetForwards(fwd_cfg, ignore_config=True).run_forward(model=model,
                                                                                         img_dir=data_s["eval_dir"],
                                                                                         n_classes=det_s["num_classes"],
                                                                                         save=False),
                                    axis=0)
            model.train()
            ap, f1 = calculate_map(predictions[predictions["s"] >= data_s["score_thr"]],
                                   data_s["gt_path"],
                                   num_classes=det_s["num_classes"],
                                   print_results=False)
            mean_ap, mean_f1 = np.nanmean(ap), np.nanmean(f1)
            print(f"mAP = {mean_ap}, mF1 = {mean_f1}.")
            lr_string = "1e{:.1}".format(np.log10(lr))
            save_path = "{}/ckpt_lr_{}_ep_{}_map_{:.3}_mf1_{:.3}.pt".format(train_s["ckpt_dir"],
                                                                            lr_string,
                                                                            epoch_i,
                                                                            mean_ap,
                                                                            mean_f1)
            save_checkpoint_weight_file(model,
                                        optimizer,
                                        epoch_i,
                                        0,
                                        losses,
                                        save_path)
        scheduler.step()


@train_ex.automain
def run_training(detector_settings,
                 data_settings,
                 train_settings):
    det_s = detector_settings
    data_s = data_settings
    train_s = train_settings

    os.makedirs(train_s["ckpt_dir"], exist_ok=True)
    dev = config_device(train_s["cpu_only"])
    model = load_yolov3_model(det_s["weight_path"],
                              dev,
                              det_s["from_ckpt"],
                              mode="train",
                              num_classes=det_s["num_classes"],
                              transfer=det_s["reset_head"])
    print(f"Loaded checkpoint from {det_s['weight_path']}.")
    print(f"Training with dropout rate {model.yolo_tail.detect1.dropout_rate}.")

    num_devs = torch.cuda.device_count()
    if num_devs > 1:
        model = torch.nn.DataParallel(model, device_ids=range(num_devs))

    train_dl = load_dataset(data_s["train_set"],
                            data_s["train_dir"],
                            data_s["train_annot"],
                            data_s["img_size"],
                            data_s["train_bs"],
                            data_s["n_cpu"],
                            True,  # Shuffle
                            data_s["augmentation"])

    for p in model.parameters():
        p.requires_grad = det_s["unfreeze_all"]

    if num_devs > 1:
        it = model.modules()
    else:
        it = model
    for layer in it.yolo_last_n_layers(det_s["last_n_layers"]):
        if det_s["reset_weights"]:
            layer.apply(init_layer_randomly)
        for p in layer.parameters():
            p.requires_grad_()

    for exponent in range(train_s["num_phases"]):
        phase(model,
              train_s["phase_epochs"],
              train_s["initial_lr"] * 10**(-exponent),
              train_dl,
              det_s,
              data_s,
              train_s)
