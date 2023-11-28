# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
Modified by Yezhen Cong, 2020
Modified by ChengJu Ho, 2023
"""

import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone_module import Pointnet2Backbone
from models.grid_conv_module import GridConv
from models.diffusion_proposal_module import DiffusionProposalModule
from models.voting_module import VotingModule
from models.loss_helper_unlabeled import trans_center, trans_size

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "pointnet2"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "models"))

from pointnet2_utils import *
from utils.nn_distance import nn_distance_topk


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def softmax(x):
    """Numpy function for softmax"""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


class VoteNet(nn.Module):
    r"""
    A deep neural network for 3D object detection with end-to-end optimizable hough voting.

    Parameters
    ----------
    num_class: int
        Number of semantics classes to predict over -- size of softmax classifier
    num_heading_bin: int
    num_size_cluster: int
    input_feature_dim: (default: 0)
        Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
        value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
    num_proposal: int (default: 128)
        Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
    vote_factor: (default: 1)
        Number of votes generated from each seed point.
    """

    def __init__(
        self,
        num_class,
        num_heading_bin,
        num_size_cluster,
        mean_size_arr,
        dataset_config,
        diffusion_config,
        input_feature_dim=0,
        num_proposal=128,
        vote_factor=1,
        sampling="seed_fps",
        query_feats="seed",
    ):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.dataset_config = dataset_config
        assert mean_size_arr.shape[0] == self.num_size_cluster
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.diffusion_config = diffusion_config
        self.num_iterative_train = diffusion_config["iterative_train"]

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Build diffusion process
        self.build_diffusion()

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        self.pnet = DiffusionProposalModule(
            num_class,
            num_heading_bin,
            num_size_cluster,
            mean_size_arr,
            num_proposal,
            sampling,
            diffusion_config,
            dataset_config,
            query_feats=query_feats,
        )

        self.grid_conv = GridConv(
            num_class,
            num_heading_bin,
            num_size_cluster,
            mean_size_arr,
            num_proposal,
            sampling,
            query_feats=query_feats,
        )

    def build_diffusion(self):

        timesteps = self.diffusion_config["timesteps"]
        sampling_timesteps = self.diffusion_config["sampling_timesteps"]

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        (timesteps,) = betas.shape

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps

        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1

        self.size_scale = self.diffusion_config["size_scale"]
        self.label_scale = self.diffusion_config["label_scale"]

        self.center_bias = self.diffusion_config["center_mean_bias"]
        self.size_bias = self.diffusion_config["size_mean_bias"]
        self.label_bias = self.diffusion_config["label_mean_bias"]

        self.center_sigma = self.center_bias / 3
        self.size_sigma = self.size_bias / 3
        self.label_sigma = self.label_bias / 3

        self.label_init_topk = 3
        self.label_assign_thres = 0.3

        self.betas = betas.cuda()
        self.alphas_cumprod = alphas_cumprod.cuda()
        self.alphas_cumprod_prev = alphas_cumprod_prev.cuda()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).cuda()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).cuda()
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod).cuda()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).cuda()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1).cuda()

    def forward_backbone(self, inputs):
        """Forward a pass through backbone but not iou branch

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formatted as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs["point_clouds"].shape[0]
        end_points = self.backbone_net(inputs["point_clouds"], end_points)

        # --------- HOUGH VOTING ---------
        xyz = end_points["fp2_xyz"]
        features = end_points["fp2_features"]
        end_points["seed_inds"] = end_points["fp2_inds"]
        end_points["seed_xyz"] = xyz
        end_points["seed_features"] = features

        xyz, features = self.vgen(xyz, features)

        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points["vote_xyz"] = xyz
        end_points["vote_features"] = features

        return end_points

    def calculate_bbox(self, end_points):
        # calculate size and center
        size_scores = end_points["size_scores"]
        size_residuals = end_points["size_residuals"]
        B, K = size_scores.shape[:2]
        mean_size_arr = self.mean_size_arr
        mean_size_arr = torch.from_numpy(
            mean_size_arr.astype(np.float32)
        ).cuda()  # (num_size_cluster,3)
        size_class = torch.argmax(size_scores, -1)  # B,num_proposal
        size_residual = torch.gather(
            size_residuals,
            2,
            size_class.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 3),
        )  # B,num_proposal,1,3
        size_residual = size_residual.squeeze(2)
        size_base = torch.index_select(mean_size_arr, 0, size_class.view(-1))
        size_base = size_base.view(B, K, 3)
        size = (size_base + size_residual) / 2  # half of the size
        size[size < 0] = 1e-6
        center = end_points["center"]

        heading_scores = end_points["heading_scores"]
        heading_class = torch.argmax(heading_scores, -1)  # B,num_proposal
        heading_residuals = end_points["heading_residuals"]
        heading_residual = torch.gather(
            heading_residuals, 2, heading_class.unsqueeze(-1)
        )  # B,num_proposal,1
        heading_residual = heading_residual.squeeze(2)
        heading = self.dataset_config.class2angle_gpu(heading_class, heading_residual)

        end_points["size"] = size
        end_points["heading"] = heading
        return center, size, heading

    def evaluate(
        self, inputs, batch_data_label, jittering=False, ema=False, iou_opt=False
    ):

        end_points = self.forward_backbone(inputs)

        xyz = end_points["vote_xyz"]
        featrues = end_points["vote_features"]

        end_points = self.ddim_sample(
            xyz,
            featrues,
            end_points,
            batch_data_label,
            ema=ema,
            jittering=jittering,
            iou_opt=iou_opt,
        )

        return end_points

    def forward(self, inputs, batch_data_label, ema_end_points=None, jittering=False):

        end_points = self.forward_backbone(inputs)

        xyz = end_points["vote_xyz"]
        features = end_points["vote_features"]

        end_points = self.prepare_targets(batch_data_label, end_points, ema_end_points)

        end_points_dict = {}

        for iter in range(self.num_iterative_train):
            end_points_dict[iter] = {}
            for key in end_points.keys():
                end_points_dict[iter][key] = end_points[key].clone()
                end_points_dict[iter][key] = end_points_dict[iter][key].cuda()

        for iter in range(self.num_iterative_train):

            end_points_dict[iter] = self.pnet(xyz, features, end_points_dict[iter])

            if jittering:
                end_points_dict[iter] = self.jitter_iou(end_points_dict[iter])

            # Update the diffusion input for the next round based on the results of this round.
            if iter < self.num_iterative_train - 1:

                pred_center = end_points_dict[iter]["center"].clone().detach()
                end_points_dict[iter + 1]["diffusion_boxes_center"] = pred_center

                size_scores = end_points_dict[iter]["size_scores"].clone().detach()
                size_residuals = (
                    end_points_dict[iter]["size_residuals"].clone().detach()
                )
                pred_size_class = torch.argmax(size_scores, -1)
                pred_size_residual = torch.gather(
                    size_residuals,
                    2,
                    pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3),
                )
                pred_size_residual = pred_size_residual.squeeze(2)

                pred_size = self.dataset_config.class2size_gpu(
                    pred_size_class, pred_size_residual
                )
                pred_size[pred_size <= 0] = 1e-6
                end_points_dict[iter + 1]["diffusion_boxes_size"] = pred_size

                pred_sem_label = (
                    end_points_dict[iter]["sem_cls_scores"].clone().detach()
                    + end_points_dict[iter]["label_cls_scores"].clone().detach()
                )
                end_points_dict[iter + 1]["diffusion_boxes_label"] = nn.Softmax(dim=2)(
                    pred_sem_label
                )

        return end_points_dict

    def prepare_targets(self, batch_data_label, end_points, ema_end_points):

        batch_size = batch_data_label["point_clouds"].shape[0]

        end_points["diffusion_boxes_center"] = torch.zeros(
            (batch_size, self.num_proposal, 3)
        ).cuda()
        end_points["diffusion_boxes_size"] = torch.zeros(
            (batch_size, self.num_proposal, 3)
        ).cuda()
        end_points["diffusion_boxes_label"] = torch.zeros(
            (batch_size, self.num_proposal, self.num_class)
        ).cuda()
        end_points["ts"] = torch.zeros((batch_size)).cuda()

        ema_pred_center = None
        ema_pred_size = None
        ema_pred_label = None
        final_mask = None

        if ema_end_points != None:

            # Prepare Pseudo Ground-truth Boxes
            ema_pred_center = ema_end_points["center"].clone().detach()

            size_scores = ema_end_points["size_scores"].clone().detach()
            size_residuals = ema_end_points["size_residuals"].clone().detach()
            pred_size_class = torch.argmax(size_scores, -1)
            pred_size_residual = torch.gather(
                size_residuals,
                2,
                pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3),
            )
            pred_size_residual = pred_size_residual.squeeze(2)

            # center and size should be transformed
            ema_pred_center = trans_center(
                ema_pred_center,
                batch_data_label["flip_x_axis"],
                batch_data_label["flip_y_axis"],
                batch_data_label["rot_mat"],
                batch_data_label["scale"],
            )
            ema_pred_size = trans_size(
                pred_size_class,
                pred_size_residual,
                batch_data_label["scale"],
                self.dataset_config,
            )

            # Prepare Pseudo Ground-truth labels
            pred_sem_label = (
                ema_end_points["sem_cls_scores"].clone().detach()
                + ema_end_points["label_cls_scores"].clone().detach()
            )
            ema_pred_label = torch.argmax(nn.Softmax(dim=2)(pred_sem_label), dim=2)

            _, _, _, final_mask = self.mask_generation(ema_end_points)

        for batch_ind in range(batch_size):

            (
                diffusion_boxes_center,
                diffusion_boxes_size,
                t,
                diffusion_boxes_label,
            ) = self.prepare_diffusion_concat(
                batch_data_label,
                batch_ind,
                end_points,
                ema_pred_center,
                ema_pred_size,
                ema_pred_label,
                final_mask,
            )

            end_points["diffusion_boxes_center"][batch_ind] = diffusion_boxes_center
            end_points["diffusion_boxes_size"][batch_ind] = diffusion_boxes_size
            end_points["diffusion_boxes_label"][batch_ind] = diffusion_boxes_label
            end_points["ts"][batch_ind] = t

        return end_points

    def prepare_diffusion_concat(
        self,
        batch_data_label,
        batch_ind,
        end_points,
        ema_pred_center,
        ema_pred_size,
        ema_pred_label,
        final_mask,
    ):

        num_of_labeled_data = torch.sum(batch_data_label["supervised_mask"])

        num_gt = batch_data_label["num_gt"][batch_ind]

        pc_xmin = batch_data_label["pc_xmin"][batch_ind]
        pc_ymin = batch_data_label["pc_ymin"][batch_ind]
        pc_zmin = batch_data_label["pc_zmin"][batch_ind]

        length = batch_data_label["length"][batch_ind]
        width = batch_data_label["width"][batch_ind]
        height = batch_data_label["height"][batch_ind]

        # Initialize -> denominator, adjusting to comply with the 3 sigma principle
        diffusion_boxes_center = (
            torch.randn(self.num_proposal, 3) * self.center_sigma + self.center_bias
        ).cuda()
        diffusion_boxes_size = (
            torch.randn(self.num_proposal, 3) * self.size_sigma + self.size_bias
        )
        diffusion_boxes_size = torch.clip(diffusion_boxes_size, min=1e-4).cuda()
        diffusion_boxes_label = (
            torch.randn(self.num_proposal, self.num_class) * self.label_sigma
            + self.label_bias
        )
        diffusion_boxes_label = torch.clip(diffusion_boxes_label, min=1e-4).cuda()

        # Labeled data: Get Ground-truth Boxes
        if (batch_ind < num_of_labeled_data) and num_gt > 0:

            fps_seed = end_points["seed_xyz"][batch_ind][: self.num_proposal].unsqueeze(
                0
            )
            gt_center = batch_data_label["center_label"][batch_ind][:num_gt].unsqueeze(
                0
            )

            k = min(self.label_init_topk, num_gt)
            _, _, dist2, ind2 = nn_distance_topk(fps_seed, gt_center, k=k)

            dist2 = dist2.squeeze(0).permute(1, 0)
            ind2 = ind2.squeeze(0).permute(1, 0)

            for gt_ind in range(num_gt):
                for j in range(k):
                    gt_box_id = ind2[gt_ind][j]
                    if j == 0:
                        diffusion_boxes_label[gt_box_id] = torch.zeros(
                            (self.num_class)
                        ).cuda()
                    elif dist2[gt_ind][j] < self.label_assign_thres:
                        diffusion_boxes_label[gt_box_id] = torch.zeros(
                            (self.num_class)
                        ).cuda()

            for gt_ind in range(num_gt):

                # Regardless of whether it exceeds the label assignment threshold or not,
                # each ground truth (GT) should be updated to at least one bounding box.
                gt_box_id = ind2[gt_ind][0]

                diffusion_boxes_center[gt_box_id] = batch_data_label["center_label"][
                    batch_ind
                ][gt_ind]
                diffusion_boxes_size[gt_box_id] = batch_data_label["size_label"][
                    batch_ind
                ][gt_ind]

                # Normalization
                diffusion_boxes_center[gt_box_id][0] = (
                    diffusion_boxes_center[gt_box_id][0] - pc_xmin
                ) / length
                diffusion_boxes_center[gt_box_id][1] = (
                    diffusion_boxes_center[gt_box_id][1] - pc_ymin
                ) / width
                diffusion_boxes_center[gt_box_id][2] = (
                    diffusion_boxes_center[gt_box_id][2] - pc_zmin
                ) / height

                diffusion_boxes_size[gt_box_id][0] = (
                    diffusion_boxes_size[gt_box_id][0] / length
                )
                diffusion_boxes_size[gt_box_id][1] = (
                    diffusion_boxes_size[gt_box_id][1] / width
                )
                diffusion_boxes_size[gt_box_id][2] = (
                    diffusion_boxes_size[gt_box_id][2] / height
                )

                label_class = batch_data_label["sem_cls_label"][batch_ind][gt_ind]
                diffusion_boxes_label[gt_box_id][label_class] = 1

                for j in range(1, k):
                    gt_box_id = ind2[gt_ind][j]
                    if dist2[gt_ind][j] < self.label_assign_thres:
                        diffusion_boxes_label[gt_box_id][label_class] = 1 - 0.1 * j

        # Unlabeled data: Get Pseudo Ground-truth Boxes
        elif batch_ind >= num_of_labeled_data:  # 處理 unlabeled data，讀取 Pseudo GT

            true_indices = torch.nonzero(final_mask[batch_ind])

            num_psuedo_gt = true_indices.shape[0]

            fps_seed = end_points["seed_xyz"][batch_ind][: self.num_proposal].unsqueeze(
                0
            )
            pseudo_gt_center = torch.gather(
                ema_pred_center[batch_ind], 0, true_indices.expand(-1, 3)
            ).unsqueeze(0)

            k = min(self.label_init_topk, num_psuedo_gt)
            _, _, dist2, ind2 = nn_distance_topk(fps_seed, pseudo_gt_center, k=k)

            dist2 = dist2.squeeze(0).permute(1, 0)
            ind2 = ind2.squeeze(0).permute(1, 0)

            for gt_ind in range(num_psuedo_gt):
                for j in range(k):
                    gt_box_id = ind2[gt_ind][j]
                    if j == 0:
                        diffusion_boxes_label[gt_box_id] = torch.zeros(
                            (self.num_class)
                        ).cuda()
                    elif dist2[gt_ind][j] < self.label_assign_thres:
                        diffusion_boxes_label[gt_box_id] = torch.zeros(
                            (self.num_class)
                        ).cuda()

            for gt_ind in range(num_psuedo_gt):

                # Regardless of whether it exceeds the label assignment threshold or not,
                # each pseudo ground truth should be updated to at least one bounding box.
                gt_box_id = ind2[gt_ind][0]

                diffusion_boxes_center[gt_box_id] = ema_pred_center[batch_ind][gt_ind]
                diffusion_boxes_size[gt_box_id] = ema_pred_size[batch_ind][gt_ind]

                # Normalization
                diffusion_boxes_center[gt_box_id][0] = (
                    diffusion_boxes_center[gt_box_id][0] - pc_xmin
                ) / length
                diffusion_boxes_center[gt_box_id][1] = (
                    diffusion_boxes_center[gt_box_id][1] - pc_ymin
                ) / width
                diffusion_boxes_center[gt_box_id][2] = (
                    diffusion_boxes_center[gt_box_id][2] - pc_zmin
                ) / height

                diffusion_boxes_size[gt_box_id][0] = (
                    diffusion_boxes_size[gt_box_id][0] / length
                )
                diffusion_boxes_size[gt_box_id][1] = (
                    diffusion_boxes_size[gt_box_id][1] / width
                )
                diffusion_boxes_size[gt_box_id][2] = (
                    diffusion_boxes_size[gt_box_id][2] / height
                )

                label_class = ema_pred_label[batch_ind][gt_ind]
                diffusion_boxes_label[gt_box_id][label_class] = 1

                for j in range(1, k):
                    gt_box_id = ind2[gt_ind][j]
                    if dist2[gt_ind][j] < self.label_assign_thres:
                        diffusion_boxes_label[gt_box_id][label_class] = 1 - 0.1 * j

        t = torch.randint(0, self.num_timesteps, (1,)).long().cuda()

        # 1) Perform calculations with the scale (SNR) ####
        diffusion_boxes_center = (diffusion_boxes_center * 2.0 - 1.0) * self.size_scale
        diffusion_boxes_size = (diffusion_boxes_size * 2.0 - 1.0) * self.size_scale
        diffusion_boxes_label = (diffusion_boxes_label * 2.0 - 1.0) * self.label_scale

        # 2) Add Noise
        diffusion_boxes_center = self.q_sample(diffusion_boxes_center, t, None)
        diffusion_boxes_size = self.q_sample(diffusion_boxes_size, t, None)
        diffusion_boxes_label = self.q_sample(diffusion_boxes_label, t, None)

        # 3) De-normalization 0~1
        diffusion_boxes_center = torch.clamp(
            diffusion_boxes_center,
            min=-0.9999 * self.size_scale,
            max=0.9999 * self.size_scale,
        )
        diffusion_boxes_size = torch.clamp(
            diffusion_boxes_size,
            min=-0.9999 * self.size_scale,
            max=0.9999 * self.size_scale,
        )
        diffusion_boxes_label = torch.clamp(
            diffusion_boxes_label,
            min=-0.9999 * self.label_scale,
            max=0.9999 * self.label_scale,
        )

        diffusion_boxes_center = ((diffusion_boxes_center / self.size_scale) + 1) / 2.0
        diffusion_boxes_size = ((diffusion_boxes_size / self.size_scale) + 1) / 2.0
        diffusion_boxes_label = ((diffusion_boxes_label / self.label_scale) + 1) / 2.0

        diffusion_boxes_center[:, 0] = diffusion_boxes_center[:, 0] * length + pc_xmin
        diffusion_boxes_center[:, 1] = diffusion_boxes_center[:, 1] * width + pc_ymin
        diffusion_boxes_center[:, 2] = diffusion_boxes_center[:, 2] * height + pc_zmin

        diffusion_boxes_size[:, 0] = diffusion_boxes_size[:, 0] * length
        diffusion_boxes_size[:, 1] = diffusion_boxes_size[:, 1] * width
        diffusion_boxes_size[:, 2] = diffusion_boxes_size[:, 2] * height

        diffusion_boxes_label = nn.Softmax(dim=1)(diffusion_boxes_label)

        return diffusion_boxes_center, diffusion_boxes_size, t, diffusion_boxes_label

    def q_sample(self, x_start, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_start).cuda()

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)

        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def jitter_iou(self, end_points, center=None, size=None, heading=None):

        if center == None:
            center, size, heading = self.calculate_bbox(end_points)

        B, origin_proposal_num = heading.shape[0:2]

        factor = 1
        center_jitter = (
            center.unsqueeze(2).expand(-1, -1, factor, -1).contiguous().view(B, -1, 3)
        )
        size_jitter = (
            size.unsqueeze(2).expand(-1, -1, factor, -1).contiguous().view(B, -1, 3)
        )
        heading_jitter = (
            heading.unsqueeze(2).expand(-1, -1, factor).contiguous().view(B, -1)
        )
        center_jitter = (
            center_jitter + size_jitter * torch.randn(size_jitter.shape).cuda() * 0.3
        )
        size_jitter = (
            size_jitter + size_jitter * torch.randn(size_jitter.shape).cuda() * 0.3
        )
        size_jitter = torch.clamp(size_jitter, min=1e-8)

        center = torch.cat([center, center_jitter], dim=1)
        size = torch.cat([size, size_jitter], dim=1)
        heading = torch.cat([heading, heading_jitter], dim=1)

        end_points = self.grid_conv(
            center.detach(), size.detach(), heading.detach(), end_points
        )
        end_points["iou_scores_jitter"] = end_points["iou_scores"][
            :, origin_proposal_num:
        ]
        end_points["iou_scores"] = end_points["iou_scores"][:, :origin_proposal_num]

        end_points["jitter_center"] = center_jitter
        end_points["jitter_size"] = size_jitter * 2
        end_points["jitter_heading"] = heading_jitter

        return end_points

    def ddim_sample(
        self,
        xyz,
        features,
        end_points,
        batch_data_label,
        ema=False,
        jittering=False,
        iou_opt=False,
    ):

        batch_size = xyz.shape[0]

        total_timesteps, sampling_timesteps, eta = (
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # Initialization
        diffused_boxes = torch.randn(batch_size, self.num_proposal, 6).cuda()
        diffused_label = torch.randn(
            batch_size, self.num_proposal, self.num_class
        ).cuda()

        if ema:
            pc_xmin = torch.unsqueeze(batch_data_label["ema_pc_xmin"], 1).expand(
                -1, self.num_proposal
            )
            pc_ymin = torch.unsqueeze(batch_data_label["ema_pc_ymin"], 1).expand(
                -1, self.num_proposal
            )
            pc_zmin = torch.unsqueeze(batch_data_label["ema_pc_zmin"], 1).expand(
                -1, self.num_proposal
            )
            length = torch.unsqueeze(batch_data_label["ema_length"], 1).expand(
                -1, self.num_proposal
            )
            width = torch.unsqueeze(batch_data_label["ema_width"], 1).expand(
                -1, self.num_proposal
            )
            height = torch.unsqueeze(batch_data_label["ema_height"], 1).expand(
                -1, self.num_proposal
            )
        else:
            pc_xmin = torch.unsqueeze(batch_data_label["pc_xmin"], 1).expand(
                -1, self.num_proposal
            )
            pc_ymin = torch.unsqueeze(batch_data_label["pc_ymin"], 1).expand(
                -1, self.num_proposal
            )
            pc_zmin = torch.unsqueeze(batch_data_label["pc_zmin"], 1).expand(
                -1, self.num_proposal
            )
            length = torch.unsqueeze(batch_data_label["length"], 1).expand(
                -1, self.num_proposal
            )
            width = torch.unsqueeze(batch_data_label["width"], 1).expand(
                -1, self.num_proposal
            )
            height = torch.unsqueeze(batch_data_label["height"], 1).expand(
                -1, self.num_proposal
            )

        x_start, x_start_label = None, None

        for time_id, (time, time_next) in enumerate(time_pairs):

            # Initialization
            time_cond = torch.full((batch_size,), time, dtype=torch.long).cuda()
            end_points["ts"] = time_cond
            tmp_diffused_boxes = diffused_boxes.clone()
            tmp_diffused_label = diffused_label.clone()

            # De-normalization
            diffused_boxes = torch.clamp(
                diffused_boxes,
                min=-0.9999 * self.size_scale,
                max=0.9999 * self.size_scale,
            )
            diffused_boxes = ((diffused_boxes / self.size_scale) + 1) / 2

            diffused_boxes[:, :, 0] = diffused_boxes[:, :, 0] * length + pc_xmin
            diffused_boxes[:, :, 1] = diffused_boxes[:, :, 1] * width + pc_ymin
            diffused_boxes[:, :, 2] = diffused_boxes[:, :, 2] * height + pc_zmin
            diffused_boxes[:, :, 3] = diffused_boxes[:, :, 3] * length
            diffused_boxes[:, :, 4] = diffused_boxes[:, :, 4] * width
            diffused_boxes[:, :, 5] = diffused_boxes[:, :, 5] * height

            end_points["diffusion_boxes_center"] = diffused_boxes[:, :, :3]
            end_points["diffusion_boxes_size"] = diffused_boxes[:, :, 3:]

            diffused_label = torch.clamp(
                diffused_label, min=-self.label_scale, max=self.label_scale
            )
            diffused_label = ((diffused_label / self.label_scale) + 1) / 2
            diffused_label = nn.Softmax(dim=2)(diffused_label)

            end_points["diffusion_boxes_label"] = diffused_label

            # Forward
            end_points_dict = {}

            for iter in range(self.num_iterative_train):
                end_points_dict[iter] = {}
                for key in end_points.keys():
                    end_points_dict[iter][key] = end_points[key].clone()
                    end_points_dict[iter][key] = end_points_dict[iter][key].cuda()

            for iter in range(self.num_iterative_train):

                # Pnet
                end_points_dict[iter] = self.pnet(xyz, features, end_points_dict[iter])

                # Update the diffusion input for the next round based on the results of this round.
                if iter < self.num_iterative_train - 1:

                    pred_center = end_points_dict[iter]["center"].clone().detach()
                    end_points_dict[iter + 1]["diffusion_boxes_center"] = pred_center

                    size_scores = end_points_dict[iter]["size_scores"].clone().detach()
                    size_residuals = (
                        end_points_dict[iter]["size_residuals"].clone().detach()
                    )
                    pred_size_class = torch.argmax(size_scores, -1)
                    pred_size_residual = torch.gather(
                        size_residuals,
                        2,
                        pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3),
                    )
                    pred_size_residual = pred_size_residual.squeeze(2)

                    pred_size = self.dataset_config.class2size_gpu(
                        pred_size_class, pred_size_residual
                    )
                    pred_size[pred_size <= 0] = 1e-6
                    end_points_dict[iter + 1]["diffusion_boxes_size"] = pred_size

                    pred_sem_label = (
                        end_points_dict[iter]["sem_cls_scores"].clone().detach()
                        + end_points_dict[iter]["label_cls_scores"].clone().detach()
                    )
                    end_points_dict[iter + 1]["diffusion_boxes_label"] = nn.Softmax(
                        dim=2
                    )(pred_sem_label)

            end_points = {}

            for key in end_points_dict[self.num_iterative_train - 1].keys():
                end_points[key] = end_points_dict[self.num_iterative_train - 1][
                    key
                ].clone()

            # using 'iou_scores' for box renewal
            center, size, heading = self.calculate_bbox(end_points)

            if iou_opt:
                center.retain_grad()
                size.retain_grad()
                if heading.requires_grad:
                    heading.retain_grad()
                end_points = self.grid_conv(center, size, heading, end_points)
            else:
                if jittering:
                    end_points = self.jitter_iou(
                        end_points, center=center, size=size, heading=heading
                    )
                else:
                    end_points = self.grid_conv(
                        center.detach(), size.detach(), heading.detach(), end_points
                    )

            # Utilize the results from Pnet and update diffusion boxes by adding noise.

            x_start = torch.zeros((batch_size, self.num_proposal, 6)).cuda()

            x_start_center = end_points["center"]

            pred_size_class = torch.argmax(end_points["size_scores"], -1)
            pred_size_residual = torch.gather(
                end_points["size_residuals"],
                2,
                pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3),
            )
            pred_size_residual = pred_size_residual.squeeze(2)
            x_start_size = self.dataset_config.class2size_gpu(
                pred_size_class, pred_size_residual
            )
            x_start_size[x_start_size <= 0] = 1e-6

            x_start_label = end_points["label_cls_scores"]

            x_start[:, :, 0] = (x_start_center[:, :, 0] - pc_xmin) / length
            x_start[:, :, 1] = (x_start_center[:, :, 1] - pc_ymin) / width
            x_start[:, :, 2] = (x_start_center[:, :, 2] - pc_zmin) / height

            x_start[:, :, 3] = x_start_size[:, :, 0] / length
            x_start[:, :, 4] = x_start_size[:, :, 1] / width
            x_start[:, :, 5] = x_start_size[:, :, 2] / height

            x_start = (x_start * 2.0 - 1.0) * self.size_scale
            x_start_label = (x_start_label * 2.0 - 1.0) * self.label_scale

            if time_next < 0:
                continue

            # Calculation with alpha
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma ** 2).sqrt()

            x_start = torch.clamp(
                x_start, min=-0.9999 * self.size_scale, max=0.9999 * self.size_scale
            )
            pred_noise = self.predict_noise_from_start(
                tmp_diffused_boxes, time_cond, x_start
            )
            noise = torch.randn_like(x_start)
            x_start = (
                x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            ).float()

            x_start_label = torch.clamp(
                x_start_label, min=-self.label_scale, max=self.label_scale
            )
            pred_noise_label = self.predict_noise_from_start(
                tmp_diffused_label, time_cond, x_start_label
            )
            noise = torch.randn_like(x_start_label)
            x_start_label = (
                x_start_label * alpha_next.sqrt() + c * pred_noise_label + sigma * noise
            ).float()

            # Box renewal -> Filter bad proposals & replenish with random boxes

            _, _, _, final_mask = self.mask_generation(end_points)

            for batch_id in range(x_start.shape[0]):

                mask = final_mask[batch_id].cuda()
                replenish_num = (mask == False).sum()

                x_start[batch_id][~mask, :] = torch.randn(replenish_num, 6).cuda()
                x_start_label[batch_id][~mask, :] = torch.randn(
                    replenish_num, self.dataset_config.num_class
                ).cuda()

            # Update diffusion boxes for next round
            diffused_boxes = x_start.clone()
            diffused_label = x_start_label.clone()

        return end_points

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def mask_generation(self, end_points):

        # obj score threshold
        pred_objectness = nn.Softmax(dim=2)(end_points["objectness_scores"])
        pos_obj = pred_objectness[:, :, 1]
        objectness_mask = pos_obj > self.diffusion_config["renewal_obj"]

        # cls score threshold
        pred_sem_cls_score = (
            end_points["sem_cls_scores"].detach()
            + end_points["label_cls_scores"].detach()
        )
        pred_sem_cls = nn.Softmax(dim=2)(pred_sem_cls_score)
        max_cls, argmax_cls = torch.max(pred_sem_cls, dim=2)
        cls_mask = max_cls > self.diffusion_config["renewal_sem_cls"]

        # iou score threshold
        iou_pred = nn.Sigmoid()(end_points["iou_scores"])
        if iou_pred.shape[2] > 1:
            iou_pred = torch.gather(iou_pred, 2, argmax_cls.unsqueeze(-1)).squeeze(
                -1
            )  # use pred semantic labels
        else:
            iou_pred = iou_pred.squeeze(-1)

        iou_threshold = self.diffusion_config["renewal_iou"]
        iou_mask = iou_pred > iou_threshold

        # Mask generation
        before_iou_mask = torch.logical_and(cls_mask, objectness_mask)
        final_mask = torch.logical_and(before_iou_mask, iou_mask)

        return pos_obj, max_cls, before_iou_mask, final_mask

    def forward_onlyiou_faster(self, end_points, center, size, heading):
        end_points = self.grid_conv(center, size, heading, end_points)
        return end_points
