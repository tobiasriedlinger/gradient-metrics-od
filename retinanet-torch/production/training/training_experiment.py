import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from sacred import Experiment

from src.access import config_device, load_retinanet_model, load_dataset, save_checkpoint_weight_file
from src.evaluation.evaluation import calculate_map

from production.get_forwards import ForwardConfig, GetForwards

train_ex = Experiment("Train Retinanet")


@train_ex.config
def train_cfg():
    # Detector Settings
    detector_settings = dict(num_classes=80,
                             weight_path="/home/riedlinger/OD/faster-rcnn-torch-lfs/weights/resnet50-19c8e357.pth",
                             from_ckpt=False,
                             reset_weights=False,
                             unfreeze_all=False,
                             reset_head=True
                             )

    data_settings = dict(train_set="coco",
                         train_dir="/home/datasets/COCO/2017/train2017",
                         train_annot="/home/datasets/COCO/2017/annotations/instances_train2017.json",
                         train_bs=8,
                         n_cpu=8,
                         img_size=800,
                         augmentation=True,
                         eval_dir="/home/datasets/COCO/2017/val2017",
                         score_thr=0.1,
                         gt_path="/home/riedlinger/dataset_ground_truth/COCO/2017/csv"
                         )

    train_settings = dict(cpu_only=False,
                          ckpt_dir="/home/riedlinger/OD/retinanet-torch-lfs/weights/coco/retrain",
                          save_epochs=2,
                          phase_epochs=3,
                          initial_lr=1E-4,
                          num_phases=2,
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
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.0001
    )
    print(f"{n_epochs} epochs with learning rate {learning_rate}.")
    scheduler = None

    for epoch_i in range(n_epochs):
        if epoch_i == 0:
            scheduler = warmup_lr_scheduler(optimizer, 500, 0.001)

        lr = getattr(optimizer, "param_groups")[-1]["lr"]

        print(f"Epoch {epoch_i}, LR = {lr}")
        cumul_losses = torch.zeros(2, dtype=torch.float).cpu()
        for batch_i, (imgs, targets) in enumerate(tqdm(train_dl)):
            with torch.autograd.detect_anomaly():
                optimizer.zero_grad()
                imgs = [i.to(dev) for i in imgs]
                targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
                loss_dict = model(imgs, targets)
                try:
                    losses = torch.stack([loss for loss in loss_dict.values()])
                    total_loss = torch.sum(losses)
                    total_loss.backward()
                except RuntimeError:
                    optimizer.zero_grad()
                    continue
                optimizer.step()
                if epoch_i == 0:
                    scheduler.step()
            cumul_losses += losses.detach().cpu()
        print("Class Loss: {:.5}, Regression Loss: {:.5}".format(cumul_losses[0].item(),
                                                                 cumul_losses[1].item()))

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
            lr_string = "1e{:.1}".format(np.log10(learning_rate))
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
    model = load_retinanet_model(det_s["weight_path"],
                                 dev,
                                 det_s["from_ckpt"],
                                 mode="train",
                                 num_classes=det_s["num_classes"],
                                 transfer=det_s["reset_head"],
                                 unfreeze=det_s["unfreeze_all"])
    print(f"Loaded checkpoint from {det_s['weight_path']}.")
    print(f"Training with dropout rate {model.head.regression_head.dropout}.")

    num_devs = torch.cuda.device_count()
    if num_devs > 1:
        model = torch.nn.DataParallel(model, device_ids=range(num_devs))

    train_dl = load_dataset(name=data_s["train_set"],
                            img_dir=data_s["train_dir"],
                            annot_dir=data_s["train_annot"],
                            img_size=data_s["img_size"],
                            batch_size=data_s["train_bs"],
                            # data_s["n_cpu"],
                            train=data_s["augmentation"])

    for exponent in range(train_s["num_phases"]):
        phase(model,
              train_s["phase_epochs"],
              train_s["initial_lr"] * 10**(-exponent),
              train_dl,
              det_s,
              data_s,
              train_s)
