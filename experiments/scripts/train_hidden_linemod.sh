#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/train.py --dataset linemod\
  --dataset_root ./datasets/linemod/Linemod_preprocessed\
  --resume_refinenet 'pose_refine_model_current.pth'
  --start_epoch 7
