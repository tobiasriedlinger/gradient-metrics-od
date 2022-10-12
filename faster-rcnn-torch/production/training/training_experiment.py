import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from sacred import Experiment

from src.access import config_device, load_frcnn_model, load_dataset, save_checkpoint_weight_file
from src.evaluation.evaluation import calculate_map

from production.training.dist_args import DistArgs
from production.get_forwards import ForwardConfig, GetForwards

train_ex = Experiment("Train Faster R-CNN")


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
                         train_bs=16,
                         n_cpu=8,
                         img_size=800,
                         augmentation=True,
                         eval_dir="/home/datasets/COCO/2017/val2017",
                         score_thr=0.1,
                         gt_path="/home/riedlinger/dataset_ground_truth/COCO/2017/csv"
                         )

    train_settings = dict(cpu_only=False,
                          n_gpu=2,
                          ckpt_dir="/home/riedlinger/OD/faster-rcnn-torch-lfs/weights/coco/retrain",
                          save_epochs=2,
                          phase_epochs=6,
                          initial_lr=1E-3,
                          l2_lambda=5E-4,
                          num_phases=2,
                          log_dir="../log",
                          verbose=False,
                          debug=False
                          )


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


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
        weight_decay=0.0005
    )
    print(f"{n_epochs} epochs with learning rate {learning_rate}.")
    scheduler = None

    for epoch_i in range(n_epochs):
        if epoch_i == 0:
            scheduler = warmup_lr_scheduler(optimizer, 500, 0.001)
        lr = getattr(optimizer, "param_groups")[-1]["lr"]

        print(f"Epoch {epoch_i}, LR = {lr}")
        cumul_losses = torch.zeros(4, dtype=torch.float)
        for _, (imgs, targets) in enumerate(tqdm(train_dl)):
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
        print("Class Loss: {:.5}, Loc-Regression Loss: {:.5}, Conf Loss: {:.5}, Loc-Proposal Loss: {:.5}".format(cumul_losses[0].item(),
                                                                            cumul_losses[1].item(),
                                                                            cumul_losses[2].item(),
                                                                            cumul_losses[3].item()))

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
                                   print_results=True)
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
    model = load_frcnn_model(det_s["weight_path"],
                             dev,
                             det_s["from_ckpt"],
                             mode="train",
                             num_classes=det_s["num_classes"],
                             transfer=det_s["reset_head"],
                             unfreeze=det_s["unfreeze_all"]
                             )
    print(f"Loaded checkpoint from {det_s['weight_path']}.")
    print(f"Training with dropout rate {model.roi_heads.box_predictor.dropout}.")

    num_devs = torch.cuda.device_count()
    if num_devs > 1:
        args = DistArgs()
        args.gpu = train_s["n_gpu"]
        args.world_size = train_s["n_gpu"]
        init_distributed_mode(args)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=list(range(num_devs)))

    train_dl = load_dataset(name=data_s["train_set"],
                            img_dir=data_s["train_dir"],
                            annot_dir=data_s["train_annot"],
                            img_size=data_s["img_size"],
                            batch_size=data_s["train_bs"],
                            train=data_s["augmentation"],
                            n_gpu=train_s["n_gpu"]
                            )

    for exponent in range(train_s["num_phases"]):
        phase(model,
              train_s["phase_epochs"],
              train_s["initial_lr"] * 10**(-exponent),
              train_dl,
              det_s,
              data_s,
              train_s
              )
