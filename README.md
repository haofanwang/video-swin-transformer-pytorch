# Video-Swin-Transformer-Pytorch
This repo is a simple usage of the official implementation ["Video Swin Transformer"](https://github.com/SwinTransformer/Video-Swin-Transformer).

![teaser](figures/teaser.png)

## Introduction

**Video Swin Transformer** is initially described in ["Video Swin Transformer"](https://arxiv.org/abs/2106.13230), which advocates an inductive bias of locality in video Transformers, leading to a better speed-accuracy trade-off compared to previous approaches which compute self-attention globally even with spatial-temporal factorization. The locality of the proposed video architecture is realized by adapting the Swin Transformer designed for the image domain, while continuing to leverage the power of pre-trained image models. Our approach achieves state-of-the-art accuracy on a broad range of video recognition benchmarks, including action recognition (`84.9` top-1 accuracy on Kinetics-400 and `86.1` top-1 accuracy on Kinetics-600 with `~20x` less pre-training data and `~3x` smaller model size) and temporal modeling (`69.6` top-1 accuracy on Something-Something v2).

## Usage

###  Installation
```
pip install -r requirements.txt
```
If this does not work, please refer to the official [install.md](docs/install.md) for installation.


### Prepare
```
git clone https://github.com/haofanwang/video-swin-transformer-pytorch.git
```
```
cd video-swin-transformer-pytorch
mkdir checkpoints && cd checkpoints
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_sthv2.pth
cd ..
```
If you want to try different models, please refer to [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) and download corresponding pretrained weight, then modify the config and pretrained weight.

### Inference
```
python example.py
```

## Acknowledgement
The code is adapted from the official [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) repository. This project is inspired by [swin-transformer-pytorch](https://github.com/berniwal/swin-transformer-pytorch), which provides the simplest code to get started.
