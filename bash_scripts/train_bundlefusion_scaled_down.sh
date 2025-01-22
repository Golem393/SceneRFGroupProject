#!/bin/bash

# Set environment variables
export BF_ROOT=/root/dataset/bundlefusion
export BF_LOG=/root/SceneRFGroupProject/logs/monoscene2/bundlefusion

# Run the training script
python scenerf/scripts/train_bundlefusion.py --bs=1 --n_gpus=1 --n_workers_per_gpu=4\
    --n_rays=1024 \
    --lr=2e-5 \
    --enable_log=True \
    --root=$BF_ROOT \
    --logdir=$BF_LOG \
    --sample_grid_size=2 \
    --n_gaussians=2 \
    --n_pts_per_gaussian=4 \
    --n_pts_uni=6 \
    --n_pts_hier=10 \
    --add_fov_hor=7 \
    --add_fov_ver=5 \
    --sphere_h=480 \
    --sphere_w=640 \
    --max_sample_depth=8 \
    --n_frames=16 \
    --frame_interval=2
