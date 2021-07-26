"""
Container class for YOLOv3 training. Re-worked to Sacred experiment in training_experiment.py which is much
easier to adapt for different data sets or settings.
"""
import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from src.access import config_device, load_yolov3_model, load_dataset, save_checkpoint_weight_file
from src.utils import init_layer_randomly
from src.yolov3_loss import yolov3_loss
from src.evaluation import calculate_map

from production.get_forwards import ForwardConfig, GetForwards


class TrainCfg(object):
    """
    Container class for training configuration.
    """

    dataset = "coco"
    img_dir = "/home/datasets/COCO/2017/train2017"
    num_classes = 20
    batch_size = 32
    n_cpu = 8
    img_size = 608
    annot_path = "/home/datasets/COCO/2017/annotations/instances_train2017.json"
    weight_path = "/home/OD/yolov3-torch-lfs/weights/yolov3_original.pt"
    cpu_only = False
    from_ckpt = True
    ckpt_dir = "/home/OD/yolov3-torch-lfs/weights/coco/retrain"
    save_epochs = 3
    total_epochs = 30
    last_n_layers = "tail"
    reset_weights = False
    use_augmentation = True
    unfreeze_all = False
    learning_rate = 1E-4
    num_phases = 4
    eval_score_thr = 0.1
    gt_path = "/home/dataset_ground_truth/COCO/2017/csv"

    log_dir = "../log"
    verbose = False
    debug = False


class YOLOv3Training(object):
    """
    Container class for YOLOv3 training pipeline.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)

        self.dev = config_device(cfg.cpu_only)
        self.model = load_yolov3_model(cfg.weight_path,
                                       self.dev,
                                       ckpt=cfg.from_ckpt,
                                       mode="train")
        print(f"Loaded checkpoint from {cfg.weight_path}.")
        print(f"Training with dropout rate {self.model.yolo_tail.detect1.dropout_rate}.")
        num_devs = torch.cuda.device_count()
        if num_devs > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(num_devs))

        self.dataloader = load_dataset(name=cfg.dataset,
                                       img_dir=cfg.img_dir,
                                       annot_dir=cfg.annot_path,
                                       img_size=cfg.img_size,
                                       batch_size=cfg.batch_size,
                                       n_cpu=cfg.n_cpu,
                                       shuffle=True,
                                       augment=cfg.use_augmentation)

    def phase(self,
              n_epochs,
              learning_rate):

        optimizer = torch.optim.Adam(
            filter(lambda par: par.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=0.0005
        )
        print(f"{n_epochs} epochs with learning rate {learning_rate}.")

        for epoch_i in range(n_epochs):
            print(f"Epoch {epoch_i}")
            cumul_losses = torch.zeros(3, dtype=torch.float).cuda()
            for batch_i, (imgs, targets, target_lengths) in enumerate(tqdm(self.dataloader)):
                with torch.autograd.detect_anomaly():
                    optimizer.zero_grad()
                    imgs = imgs.to(self.dev)
                    targets = targets.to(self.dev)
                    target_lengths = target_lengths.to(self.dev)
                    result = self.model(imgs)
                    try:
                        losses = yolov3_loss(result, targets, target_lengths, self.cfg.img_size, True)
                        losses[0].backward()
                    except RuntimeError:
                        optimizer.zero_grad()
                        continue
                    optimizer.step()
                cumul_losses += torch.stack(losses[1:]).detach()
            print("Loc Loss: {:.5}, Conf Loss: {:.5}, Class Loss: {:.5}".format(cumul_losses[0].item(),
                                                                                cumul_losses[1].item(),
                                                                                cumul_losses[2].item()))

            if (epoch_i + 1) % self.cfg.save_epochs == 0:
                fwd_cfg = ForwardConfig()
                predictions = pd.concat(GetForwards(fwd_cfg).run_forward(model=self.model,
                                                                         n_classes=self.cfg.num_classes,
                                                                         save=False),
                                        axis=0)
                self.model.train()
                ap, f1 = calculate_map(predictions[predictions["s"] >= self.cfg.eval_score_thr],
                                       self.cfg.gt_path,
                                       num_classes=self.cfg.num_classes)
                mean_ap, mean_f1 = np.nanmean(ap), np.nanmean(f1)
                print(f"mAP = {mean_ap}, mF1 = {mean_f1}.")
                lr_string = "1e{:.1}".format(np.log10(learning_rate))
                save_path = "{}/ckpt_lr_{}_ep_{}_map_{:.3}_mf1_{:.3}.pt".format(self.cfg.ckpt_dir,
                                                                                lr_string,
                                                                                epoch_i,
                                                                                mean_ap,
                                                                                mean_f1)
                save_checkpoint_weight_file(self.model, optimizer, epoch_i, 0, losses, save_path)

    def run_training(self):
        cfg = self.cfg

        for p in self.model.parameters():
            p.requires_grad = cfg.unfreeze_all
        for layer in self.model.yolo_last_n_layers(cfg.last_n_layers):
            if self.cfg.reset_weights:
                layer.apply(init_layer_randomly)
            for p in layer.parameters():
                p.requires_grad_()

        for exponent in range(cfg.num_phases):
            self.phase(cfg.total_epochs, cfg.learning_rate * 10**(-exponent))


if __name__ == "__main__":
    config = TrainCfg()
    YOLOv3Training(config).run_training()
