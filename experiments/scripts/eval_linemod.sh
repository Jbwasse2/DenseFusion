#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_linemod.py --dataset_root ./datasets/linemod/Linemod_preprocessed\
  --model trained_models/linemod/pose_model_5_0.015415874099140027.pth\
