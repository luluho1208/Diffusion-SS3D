# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Written by ChengJu Ho, 2023

import os
import sys
import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "pointnet2"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "models"))

from pointnet2_utils import *
import pytorch_utils as pt_utils
from pointnet2 import pointnet2_utils

# help to filter points with bbox
def in_hull(p, hull):
    from scipy.spatial import Delaunay

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    """pc: (N,3), box3d: (8,3)"""
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return box3d_roi_inds


def my_compute_box_3d(center, size):
    l, w, h = size
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, -w, -w, w, w, -w, -w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    return np.transpose(corners_3d)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def decode_scores(
    net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr
):
    net_transposed = net.transpose(2, 1)  # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:, :, 0:2]
    end_points["objectness_scores"] = objectness_scores

    base_xyz = end_points["aggregated_vote_xyz"]  # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:, :, 2:5]  # (batch_size, num_proposal, 3)
    end_points["center"] = center

    heading_scores = net_transposed[:, :, 5 : 5 + num_heading_bin]
    heading_residuals_normalized = net_transposed[
        :, :, 5 + num_heading_bin : 5 + num_heading_bin * 2
    ]
    end_points["heading_scores"] = heading_scores  # Bxnum_proposalxnum_heading_bin
    end_points[
        "heading_residuals_normalized"
    ] = heading_residuals_normalized  # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points["heading_residuals"] = heading_residuals_normalized * (
        np.pi / num_heading_bin
    )  # Bxnum_proposalxnum_heading_bin

    size_scores = net_transposed[
        :, :, 5 + num_heading_bin * 2 : 5 + num_heading_bin * 2 + num_size_cluster
    ]
    size_residuals_normalized = net_transposed[
        :,
        :,
        5
        + num_heading_bin * 2
        + num_size_cluster : 5
        + num_heading_bin * 2
        + num_size_cluster * 4,
    ].view(
        [batch_size, num_proposal, num_size_cluster, 3]
    )  # Bxnum_proposalxnum_size_clusterx3
    end_points["size_scores"] = size_scores
    size_residuals_normalized = F.softplus(size_residuals_normalized) - 1

    end_points["size_residuals_normalized"] = size_residuals_normalized
    end_points["size_residuals"] = size_residuals_normalized * torch.from_numpy(
        mean_size_arr.astype(np.float32)
    ).cuda().unsqueeze(0).unsqueeze(0)

    sem_cls_scores = net_transposed[
        :, :, 5 + num_heading_bin * 2 + num_size_cluster * 4 :
    ]  # Bxnum_proposalx10
    end_points["sem_cls_scores"] = sem_cls_scores
    return end_points


class DiffusionVoteAggregation(nn.Module):
    def __init__(
        self,
        npoint,
        nsample,
        mlp,
        seed_feat_dim,
        dataset_config,
        diffusion_config,
        bn=True,
        pooling="max",
        sample_uniformly=False,
    ):
        super().__init__()

        # Initialization

        self.npoint = npoint
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.seed_feat_dim = seed_feat_dim
        self.dataset_config = dataset_config
        self.diffusion_config = diffusion_config

        self.dmodel = 128
        self.dmodel_label = 256

        # Diffusion time embbeding for center and size

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.dmodel),
            nn.Linear(self.dmodel, self.dmodel * 4),
            nn.GELU(),
            nn.Linear(self.dmodel * 4, self.dmodel * 4),
        )

        self.block_time_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dmodel * 4, self.dmodel * 2)
        )

        # Diffusion time embbeding for label

        self.time_mlp_label = nn.Sequential(
            SinusoidalPositionEmbeddings(self.dmodel_label),
            nn.Linear(self.dmodel_label, self.dmodel_label * 4),
            nn.GELU(),
            nn.Linear(self.dmodel_label * 4, self.dmodel_label * 4),
        )

        self.block_time_mlp_label = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dmodel_label * 4, self.dmodel_label * 2)
        )

        # mlp for center and size

        mlp_spec = copy.deepcopy(mlp)
        mlp_spec[0] = mlp_spec[0] + 3  # for xyz
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)

        # mlp for label

        mlp_spec_label = [
            self.dmodel + self.dataset_config.num_class,
            self.dmodel_label,
            self.dmodel_label,
            self.dmodel_label,
        ]
        self.mlp_module_label = pt_utils.SharedMLP(mlp_spec_label, bn=bn)

    def forward(self, xyz, features, end_points, inds=None):

        batch_size = xyz.shape[0]

        diffused_boxes_sizes = end_points["diffusion_boxes_size"]
        diffused_boxes_label = end_points["diffusion_boxes_label"]

        sample_index = torch.zeros([batch_size, self.npoint, self.nsample])

        if inds is None:
            inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        else:
            assert inds.shape[1] == self.npoint

        xyz_trans = xyz.transpose(1, 2).contiguous()

        new_xyz = (
            pointnet2_utils.gather_operation(xyz_trans, inds)
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )

        # Use the size assigned to each proposal box to collect the points.

        for batch_id in range(batch_size):

            xyz_numpy = xyz[batch_id].detach().cpu().numpy()

            for proposed_bbox_id in range(self.npoint):

                center_numpy = (
                    new_xyz[batch_id][proposed_bbox_id].detach().cpu().numpy()
                )
                size_numpy = (
                    diffused_boxes_sizes[batch_id][proposed_bbox_id]
                    .detach()
                    .cpu()
                    .numpy()
                    / 2
                )

                try:
                    corners_3d = my_compute_box_3d(center_numpy, size_numpy)
                    index = extract_pc_in_box3d(xyz_numpy, corners_3d)
                    index = np.nonzero(index)[0]

                    index = np.random.choice(index, size=self.nsample, replace=True)
                    index = torch.tensor(index, dtype=torch.int).cuda()
                    sample_index[batch_id][proposed_bbox_id] = index
                except:
                    print("QJ ERROR", center_numpy, size_numpy)

        sample_index = sample_index.cuda().type(torch.cuda.IntTensor)

        # Aggregation nsample points

        grouped_xyz = pointnet2_utils.grouping_operation(xyz_trans, sample_index)
        grouped_xyz = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)

        grouped_features = pointnet2_utils.grouping_operation(features, sample_index)
        grouped_features = torch.cat([grouped_xyz, grouped_features], dim=1)

        new_features = self.mlp_module(grouped_features)

        if self.pooling == "max":
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )

        elif self.pooling == "avg":
            new_features = F.avg_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )

        label_feature = new_features.clone()
        new_features = new_features.squeeze(-1)

        time_emb = self.time_mlp(end_points["ts"])
        end_points["time_emb"] = time_emb

        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, self.npoint, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.reshape(-1, self.npoint)
        shift = shift.reshape(-1, self.npoint)

        new_features = new_features.view(-1, self.npoint)
        new_features = new_features * (scale + 1) + shift
        new_features = new_features.view(batch_size, -1, self.npoint)

        diffused_boxes_label_trans = (
            diffused_boxes_label.transpose(1, 2).contiguous().unsqueeze(-1)
        )
        label_feature = torch.cat([diffused_boxes_label_trans, label_feature], dim=1)
        label_feature = self.mlp_module_label(label_feature)
        label_feature = label_feature.squeeze(-1)

        time_emb_label = self.time_mlp_label(end_points["ts"])
        end_points["time_emb_label"] = time_emb_label

        scale_shift = self.block_time_mlp_label(time_emb_label)
        scale_shift = torch.repeat_interleave(scale_shift, self.npoint, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.reshape(-1, self.npoint)
        shift = shift.reshape(-1, self.npoint)

        label_feature = label_feature.view(-1, self.npoint)
        label_feature = label_feature * (scale + 1) + shift
        label_feature = label_feature.view(batch_size, -1, self.npoint)

        return new_xyz, new_features, label_feature, inds


class DiffusionProposalModule(nn.Module):
    def __init__(
        self,
        num_class,
        num_heading_bin,
        num_size_cluster,
        mean_size_arr,
        num_proposal,
        sampling,
        diffusion_config,
        dataset_config,
        seed_feat_dim=256,
        query_feats="seed",
    ):
        super().__init__()

        # Initialization
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.query_feats = query_feats
        self.diffusion_config = diffusion_config
        self.dataset_config = dataset_config
        self.nsample = 32

        # Custom vote clustering
        self.diffusion_vote_aggregation = DiffusionVoteAggregation(
            npoint=self.num_proposal,
            nsample=self.nsample,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            seed_feat_dim=self.seed_feat_dim,
            dataset_config=self.dataset_config,
            diffusion_config=self.diffusion_config,
        )

        # For votenet conv

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(
            128, 2 + 3 + num_heading_bin * 2 + num_size_cluster * 4 + self.num_class, 1
        )
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

        # For diffusion label conv

        self.label_conv1 = torch.nn.Conv1d(256, 256, 1)
        self.label_conv2 = torch.nn.Conv1d(256, 256, 1)
        self.label_conv3 = torch.nn.Conv1d(256, self.num_class, 1)
        self.label_bn1 = torch.nn.BatchNorm1d(256)
        self.label_bn2 = torch.nn.BatchNorm1d(256)

    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        # xyz & feature aggregation
        if self.sampling == "seed_fps":
            # FPS on seed and choose the votes corresponding to the seeds
            sample_inds = pointnet2_utils.furthest_point_sample(
                end_points["seed_xyz"], self.num_proposal
            )
            xyz, features, label_feature, _ = self.diffusion_vote_aggregation(
                xyz, features, end_points, inds=sample_inds
            )
        else:
            print("Unknown sampling strategy: %s. Exiting!" % (self.sampling))
            exit()

        end_points["aggregated_vote_xyz"] = xyz
        end_points["aggregated_vote_inds"] = sample_inds

        # Proposal generation
        net = F.relu(self.bn1(self.conv1(features)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.conv3(
            net
        )  # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        end_points = decode_scores(
            net,
            end_points,
            self.num_class,
            self.num_heading_bin,
            self.num_size_cluster,
            self.mean_size_arr,
        )

        # Label cls generation

        net_label = F.relu(self.label_bn1(self.label_conv1(label_feature)))
        net_label = F.relu(self.label_bn2(self.label_conv2(net_label)))
        net_label = self.label_conv3(net_label)  # (batch_size, num_class, num_proposal)
        net_label = net_label.transpose(2, 1)

        end_points["label_cls_scores"] = net_label

        return end_points
