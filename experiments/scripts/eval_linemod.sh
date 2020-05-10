#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_linemod.py --dataset_root ./datasets/linemod/Linemod_preprocessed\
  --model trained_models/linemod/pose_model_7_0.01272672920826872.pth
  --refine_model trained_models/linemod/pose_refine_model_current.pth