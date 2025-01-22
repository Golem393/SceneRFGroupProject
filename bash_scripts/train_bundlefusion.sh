#!/bin/bash

# Set environment variables
export BF_ROOT=/root/SceneRFGroupProject/dataset/bundlefusion
export BF_LOG=/root/SceneRFGroupProject/logs/monoscene2/bundlefusion

# Run the training script
python scenerf/scripts/train_bundlefusion.py --bs=1 --n_gpus=1 \
    --n_rays=720 --lr=2e-5 \
    --enable_log=True \
    --root=$BF_ROOT \
    --logdir=$BF_LOG \