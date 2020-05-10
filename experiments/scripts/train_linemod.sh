#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/train.py --dataset linemod\
  --dataset_root ./datasets/linemod/Linemod_preprocessed\
  --resume_posenet pose_model_4_0.016889752443352838.pth\
  --start_epoch 5
