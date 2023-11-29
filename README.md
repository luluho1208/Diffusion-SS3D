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

Install TensorFlow (for TensorBoard) (We tested this repo with TensorFlow 2.9.1.)
```
pip install tensorflow
```

Install and register for [wandb](https://wandb.ai/site) [QuickStart](https://docs.wandb.ai/quickstart)
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
Please download the ScanNet data following the [README](https://github.com/luluho1208/Diffusion-SS3D/tree/main/scannet/README.md) in scannet folder.

## Download Pre-trained and Trained Models
We provide the pre-trained models of ScanNet 5%: [Link](https://drive.google.com/file/d/1K1l8TbGKnXD4bOzdHVoxQ1775yACQK9e/view?usp=sharing)
We provide the trained model of ScanNet 5%: [Link](https://drive.google.com/file/d/1g12CrVly8B1xXit7iEw9Et8_vokNc5DG/view?usp=sharing)

## Pre-training
```
sh run_pretrain.sh <GPU_ID> <LOG_DIR> <DATA_RATIO> <LABELED_LIST>
```
For example:
```
sh run_pretrain.sh 0 results/pretrain 0.05 scannetv2_train_0.05.txt
```

## Training
```
sh run_train.sh <GPU_ID> <LOG_DIR> <DATA_RATIO> <LABELED_LIST> <PRETRAINED_CKPT>
```
For example:
```
sh run_train.sh 0 results/train 0.05 scannetv2_train_0.05.txt results/pretrain/best_checkpoint_sum.tar
```

## Evaluation
```
sh run_eval.sh <GPU_ID> <LOG_DIR> <DATA_RATIO> <LABELED_LIST> <CKPT>
```
For example:
```
sh run_eval.sh 0 results/eval 0.05 scannetv2_train_0.05.txt results/train/best_checkpoint_sum.tar
```

## Evaluation with IoU Optimization
```
sh run_eval_opt.sh <GPU_ID> <LOG_DIR> <DATA_RATIO> <LABELED_LIST> <CKPT> <OPT_RATE>
```
The number of steps (of optimization) is by default 10.

## Acknowledgements
Our implementation uses code from the following repositories:
- [Deep Hough Voting for 3D Object Detection in Point Clouds](https://github.com/facebookresearch/votenet)
- [3DIoUMatch](https://github.com/THU17cyz/3DIoUMatch)
- [Pointnet2/Pointnet++ PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet)
