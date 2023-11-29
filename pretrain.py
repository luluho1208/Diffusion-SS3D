""" Pre-training function for Diffusion-SS3D

Modified by ChengJu Ho, 2023
Based on: VoteNet, SESS, 3DIoUMatch, DiffusionDet
"""

import os
import sys
import wandb
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.dump_helper import dump_results
from models.loss_helper_labeled import get_labeled_loss
from models.votenet_iou_branch import VoteNet as Detector
from ap_helper import APCalculator, parse_predictions, parse_groundtruths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "pointnet2"))
sys.path.append(os.path.join(ROOT_DIR, "models"))

from pytorch_utils import BNMomentumScheduler

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="scannet", help="Dataset name.")
parser.add_argument("--checkpoint_path", default=None, help="Model checkpoint path [default: None]")
parser.add_argument("--log_dir", default="temp", help="Dump dir to save model checkpoint [default: log]")
parser.add_argument("--dump_dir", default=None, help="Dump dir to save sample outputs [default: None]")
parser.add_argument("--num_point", type=int, default=40000, help="Point Number [default: 40000]")
parser.add_argument("--num_target", type=int, default=128, help="Proposal number [default: 128]")
parser.add_argument("--vote_factor", type=int, default=1, help="Vote factor [default: 1]")
parser.add_argument("--ap_iou_thresh", type=float, default=0.25, help="AP IoU threshold [default: 0.25]")
parser.add_argument("--max_epoch", type=int, default=901, help="Epoch to run [default: 901]")
parser.add_argument("--batch_size", type=int, default=8, help="Batch Size during training [default: 8]")
parser.add_argument("--learning_rate", type=float, default=0.005, help="Initial learning rate [default: 0.005]")
parser.add_argument("--weight_decay", type=float, default=0, help="Optimization L2 weight decay [default: 0]")
parser.add_argument("--bn_decay_step", type=int, default=20, help="Period of BN decay (in epochs) [default: 20]")
parser.add_argument("--bn_decay_rate", type=float, default=0.5, help="Decay rate for BN decay [default: 0.5]")
parser.add_argument("--lr_decay_steps", default="400, 600, 800", help="When to decay the learning rate (in epochs) [default: 400,600,800]")
parser.add_argument("--lr_decay_rates", default="0.1, 0.1, 0.1", help="Decay rates for lr decay [default: 0.1,0.1,0.1]")
parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing log and dump folders.")
parser.add_argument("--dump_results", action="store_true", help="Dump results.")
parser.add_argument("--iou_weight", type=float, default=1.0)
parser.add_argument("--labeled_sample_list", default="scannetv2_train_0.05.txt", type=str)
parser.add_argument("--print_interval", type=int, default=10, help="batch inverval to print loss")
parser.add_argument("--eval_interval", type=int, default=10, help="epoch inverval to evaluate model")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--data_ratio", type=float, default=0.05, help="data_ratio")
parser.add_argument("--iterative_train", type=int, default=2, help="Number of forward training passes.")
parser.add_argument("--size_snr_scale", type=float, default=2.0, help="snr_scale for diffusion size.")
parser.add_argument("--label_snr_scale", type=float, default=4.0, help="snr_scale for diffusion label.")
parser.add_argument("--timesteps", type=int, default=1000, help="timesteps")
parser.add_argument("--sampling_timesteps", type=int, default=2, help="Num of sampling timesteps for diffusion process.")
parser.add_argument("--use_wandb", action="store_true")
FLAGS = parser.parse_args()
################################ END ################################

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
print("************************** GLOBAL CONFIG BEG **************************")

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(",")]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(",")]
assert len(LR_DECAY_STEPS) == len(LR_DECAY_RATES)

LOG_DIR = FLAGS.log_dir
DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, LOG_DIR)
DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, "checkpoint.tar")
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH
FLAGS.DUMP_DIR = DUMP_DIR

print(FLAGS)

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print("Log folder %s already exists. Are you sure to overwrite? (Y/N)" % (LOG_DIR))
    c = input()
    if c == "n" or c == "N":
        print("Exiting..")
        exit()
    elif c == "y" or c == "Y":
        print("Overwrite the files in the log and dump folers...")
        os.system("rm -r %s %s" % (LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

if not os.path.exists(DUMP_DIR): os.makedirs(DUMP_DIR)
    
LOG_FOUT = open(os.path.join(LOG_DIR, "log_train.txt"), "a")
LOG_FOUT.write(str(FLAGS) + "\n")
PERFORMANCE_FOUT = open(os.path.join(LOG_DIR, "best.txt"), "w")


def log_string(out_str):
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    print(out_str)

# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create Dataset and Dataloader
if FLAGS.dataset == "scannet":
    from scannet.scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
    from scannet.model_util_scannet import ScannetDatasetConfig

    DATASET_CONFIG = ScannetDatasetConfig()
    TRAIN_DATASET = ScannetDetectionDataset(
        "train",
        num_points=NUM_POINT,
        augment=True,
        use_color=FLAGS.use_color,
        use_height=(not FLAGS.no_height),
        labeled_sample_list=FLAGS.labeled_sample_list,
    )
    TEST_DATASET = ScannetDetectionDataset(
        "val",
        num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color,
        use_height=(not FLAGS.no_height),
    )

else:
    print("Unknown dataset %s. Exiting..." % (FLAGS.dataset))
    exit(-1)

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, worker_init_fn=my_worker_init_fn)
TEST_DATALOADER  = DataLoader(TEST_DATASET,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, worker_init_fn=my_worker_init_fn)

if FLAGS.use_wandb:
    wandb.init(project=("Diffusion-SS3D_pretrain_" + str(FLAGS.data_ratio)), entity="dev", name=os.path.basename(LOG_DIR))
    wandb.config = {"learning_rate": BASE_LEARNING_RATE, "epochs": MAX_EPOCH, "batch_size": BATCH_SIZE}


# Used for AP calculation
CONFIG_DICT = {
    "remove_empty_box": False,
    "use_3d_nms": True,
    "nms_iou": 0.25,
    "use_old_type_nms": False,
    "cls_nms": True,
    "use_iou_for_nms": False,
    "per_class_proposal": True,
    "conf_thresh": 0.05,
    "iou_weight": FLAGS.iou_weight,
    "dataset_config": DATASET_CONFIG,
}

# Used for Diffusion Process
DIFFUSION_CONFIG = {
    "dataset": FLAGS.dataset,
    "size_scale": FLAGS.size_snr_scale,
    "label_scale": FLAGS.label_snr_scale,
    "timesteps": FLAGS.timesteps,
    "sampling_timesteps": FLAGS.sampling_timesteps,
    "iterative_train": FLAGS.iterative_train,
    "renewal_obj": 0.9,
    "renewal_iou": 0.25,
    "renewal_sem_cls": 0.9,
    "size_mean_bias": 0.25,
    "center_mean_bias": 0.5,
    "label_mean_bias": 1 / 18,
    "label_loss_weight": 0.1,
}

for key in CONFIG_DICT.keys():
    if key != "dataset_config":
        log_string(key + ": " + str(CONFIG_DICT[key]))

for key in DIFFUSION_CONFIG.keys():
    log_string(key + ": " + str(DIFFUSION_CONFIG[key]))

# Init the model and optimzier
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color) * 3 + int(not FLAGS.no_height) * 1

net = Detector(
    num_class=DATASET_CONFIG.num_class,
    num_heading_bin=DATASET_CONFIG.num_heading_bin,
    num_size_cluster=DATASET_CONFIG.num_size_cluster,
    mean_size_arr=DATASET_CONFIG.mean_size_arr,
    dataset_config=DATASET_CONFIG,
    num_proposal=FLAGS.num_target,
    input_feature_dim=num_input_channel,
    vote_factor=FLAGS.vote_factor,
    diffusion_config=DIFFUSION_CONFIG,
)

net.to(device)

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# Load checkpoint if there is any
it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint["model_state_dict"])

    if FLAGS.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch"s BN momentum (default 0.1)= 1 - tensorflow"s BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE ** (int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch - 1)
if FLAGS.resume:
    bnm_scheduler.step(start_epoch)

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i, lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

print("************************** GLOBAL CONFIG END **************************")
# ------------------------------------------------------------------------- GLOBAL CONFIG END

AP_IOU_THRESHOLDS = [0.25, 0.5]


def evaluate_one_epoch():
    stat_dict = {}  # collect statistics
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) for iou_thresh in AP_IOU_THRESHOLDS]
    net.eval()  # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        inputs = {"point_clouds": batch_data_label["point_clouds"]}
        with torch.no_grad():
            end_points = net.evaluate(inputs, batch_data_label, jittering=False, ema=False)

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]

        _, end_points = get_labeled_loss(end_points, DATASET_CONFIG, DIFFUSION_CONFIG)

        # Accumulate statistics and print out
        for key in end_points:
            if "loss" in key or "acc" in key or "ratio" in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

        # Dump evaluation results for visualization
        if FLAGS.dump_results and batch_idx == 0 and EPOCH_CNT % 10 == 0:
            dump_results(end_points, DUMP_DIR, DATASET_CONFIG)

    # Log statistics
    for key in sorted(stat_dict.keys()):
        log_string("eval mean %s: %f" % (key, stat_dict[key] / (float(batch_idx + 1))))

    map = []
    # Evaluate average precision
    for i, ap_calculator in enumerate(ap_calculator_list):
        print("-" * 10, "iou_thresh: %f" % (AP_IOU_THRESHOLDS[i]), "-" * 10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string("eval %s: %f" % (key, metrics_dict[key]))
        map.append(metrics_dict["mAP"])

    mean_loss = stat_dict["loss"] / float(batch_idx + 1)

    for key in sorted(stat_dict.keys()):
        stat_dict[key] /= len(TEST_DATALOADER)

    return mean_loss, map, stat_dict


def train_one_epoch():
    stat_dict = {}  # collect statistics
    epoch_stat_dict = {}
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step()  # decay BN momentum
    net.train()  # set model to training mode

    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {"point_clouds": batch_data_label["point_clouds"]}
        end_points_dict = net(inputs, batch_data_label, ema_end_points=None, jittering=True)

        total_loss = 0
        
        for iter_id in range(FLAGS.iterative_train):
            # Compute loss and gradients, update parameters.
            for key in batch_data_label:
                assert key not in end_points_dict[iter_id]
                end_points_dict[iter_id][key] = batch_data_label[key]

            loss, end_points_dict[iter_id] = get_labeled_loss(end_points_dict[iter_id], DATASET_CONFIG, DIFFUSION_CONFIG)

            total_loss = total_loss + loss

        total_loss.backward()
        optimizer.step()

        # Accumulate statistics and print out
        iter_id = FLAGS.iterative_train - 1
        for key in end_points_dict[iter_id]:
            if "loss" in key or "acc" in key or "ratio" in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points_dict[iter_id][key].item()
                
                if key not in epoch_stat_dict: epoch_stat_dict[key] = 0
                epoch_stat_dict[key] += end_points_dict[iter_id][key].item()

        batch_interval = FLAGS.print_interval
        if (batch_idx + 1) % batch_interval == 0:
            log_string(" ---- batch: %03d ----" % (batch_idx + 1))

            for key in sorted(stat_dict.keys()):
                log_string("mean %s: %f" % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0

    for key in epoch_stat_dict.keys():
        epoch_stat_dict[key] /= len(TRAIN_DATALOADER)

    return epoch_stat_dict


BEST_MAP = [0.0, 0.0]


def wandb_log(epoch, train_stat_dict, eval_stat_dict, map, best_map):

    to_log = {}

    to_log["epoch"] = epoch
    to_log["learning rate"] = get_current_lr(epoch)

    for key, value in train_stat_dict.items():
        to_log["train_" + key] = value

    if eval_stat_dict != None:
        for key, value in eval_stat_dict.items():
            to_log["eval_" + key] = value

        to_log["mAP 0.25"] = map[0]
        to_log["mAP 0.5"] = map[1]

    to_log["Best mAP 0.25"] = best_map[0]
    to_log["Best mAP 0.5"] = best_map[1]
    to_log["Best mAP (0.25 + 0.5)"] = best_map[0] + best_map[1]

    wandb.log(to_log)


def train(start_epoch):
    global EPOCH_CNT
    loss = 0
    global BEST_MAP
    EPOCH_CNT = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string("**** EPOCH %03d ****" % (epoch))
        log_string("Current learning rate: %f" % (get_current_lr(epoch)))
        log_string("Current BN decay momentum: %f" % (bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))

        # in numpy 1.18.5 this actually sets `np.random.get_state()[1][0]` to default value
        # so the test data is consistent as the initial seed is the same
        np.random.seed()

        train_stat_dict = train_one_epoch()
        map = [0.0, 0.0]
        eval_stat_dict = None

        if EPOCH_CNT % FLAGS.eval_interval == 0 and EPOCH_CNT > 0:
            loss, map, eval_stat_dict = evaluate_one_epoch()
            if map[0] + map[1] > BEST_MAP[0] + BEST_MAP[1]:
                BEST_MAP = map
                save_dict = {
                    "epoch": epoch + 1,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                }
                try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                    save_dict["model_state_dict"] = net.module.state_dict()
                except:
                    save_dict["model_state_dict"] = net.state_dict()
                torch.save(save_dict, os.path.join(LOG_DIR, "best_checkpoint_sum.tar"))
            
            PERFORMANCE_FOUT.write(
                "epoch: "
                + str(EPOCH_CNT)
                + "\n"
                + "best: "
                + str(BEST_MAP[0].item())
                + ", "
                + str(BEST_MAP[1].item())
                + "\n"
            )
            
            PERFORMANCE_FOUT.flush()

        if FLAGS.use_wandb:
            wandb_log(epoch, train_stat_dict, eval_stat_dict, map, BEST_MAP)


if __name__ == "__main__":
    train(start_epoch)