# Diffusion-SS3D: Diffusion Model for Semi-supervised 3D Object Detection

## Introduction

This is the code release of our NeurIPS 2023 paper ["Diffusion-SS3D: Diffusion Model for Semi-supervised 3D Object Detection"](https://nips.cc/virtual/2023/poster/71274).

In this repository, we provide an implementation (with Pytorch) based on [VoteNet](https://github.com/facebookresearch/votenet), [3DIoUMatch](https://github.com/THU17cyz/3DIoUMatch) and [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet) with some modification, as well as the training and evaluation scripts on ScanNet.

## Installation
This repo is tested under the following environment:
- Python 3.7.16
- NumPy 1.21.5
- pytorch 1.10.1, cuda 11.3, torchvision 0.11.2
- Pointnet2 from [Pointnet2/Pointnet++ PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
- OpenPCDet from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

You can follow the steps below to install the above dependencies:
```
# Create and activate virtualenv
conda create -n myenv python=3.7.16
conda activate myenv
```

Install NumPy
```
pip install numpy==1.21.5
```

Install PyTorch according to your CUDA version.
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Install and register for [wandb](https://wandb.ai/site)
```
pip install wandb
```

Compile the CUDA code for PointNet++, which is used in the backbone network. If you have any probelm about this part, you can refer to [Pointnet2/Pointnet++ PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch#building-only-the-cuda-kernels)
```
cd pointnet2
python setup.py install
```

Compile the CUDA code for general 3D IoU calculation in [OpenPCDet](https://github.com/open-mmlab/OpenPCDet):
```
cd OpenPCDet
python setup.py develop
```

Install dependencies:
```
pip install -r requirements.txt
```

## Dataset
### ScanNet
Please download the ScanNet data following the [README](https://github.com/nomiaro/OPA/blob/main/scannet/README.md) in scannet folder.

## Download Pre-trained and Trained Models
We provided the pre-trained models of ScanNet 5%:
We also provided the trained model of ScanNet 5%:

## Pre-training
Pre-train with script.
```
sh run_pretrain.sh <GPU_ID> <LOG_DIR> <LABELED_LIST>
```
For example:
```
sh run_pretrain.sh 0 results/pretrain scannetv2_train_0.05.txt
```

## Training
After pre-training or downloading the checkpoint, you can train with script as follow.
```
sh run_train.sh <GPU_ID> <LOG_DIR> <LABELED_LIST> <PRETRAINED_DETECOR_CKPT>
```
For example:
```
sh run_train.sh 0 results/train scannetv2_train_0.05.txt results/pretrain/best_checkpoint_sum.tar
```

## Evaluation
After training or downloading the checkpoint, you can evaluate with script as follow.
```
sh run_eval_opt.sh <GPU_ID> <LOG_DIR> <LABELED_LIST> <CKPT> <OPT_RATE>
```
The number of steps (of optimization) is by default 10.

## Acknowledgements
Our implementation uses code from the following repositories:
- [Deep Hough Voting for 3D Object Detection in Point Clouds](https://github.com/facebookresearch/votenet)
- [3DIoUMatch](https://github.com/THU17cyz/3DIoUMatch)
- [Pointnet2/Pointnet++ PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet)
