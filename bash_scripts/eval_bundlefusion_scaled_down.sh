#!/bin/bash

# Set environment variables
export BF_ROOT=/root/dataset/bundlefusion
export BF_LOG=/root/SceneRFGroupProject/logs/monoscene2/bundlefusion
export EVAL_SAVE_DIR=/root/SceneRFGroupProject/evaluation/u_net_attn/eval
export RECON_SAVE_DIR=/root/SceneRFGroupProject/evaluation/u_net_attn/recon
export MODEL_PATH=/root/SceneRFGroupProject/logs/monoscene2/bundlefusion/attn_exp_lr2e-05_1024rays_b7_nGaus2_nPtsPerGaus4_std0.1_SOMSigma0.02_sphere640x480_addfov7x5_nFrames16_frameInterval2/checkpoints/last.ckpt

# Novel depths synthesis on Bundlefusion
python scenerf/scripts/evaluation/save_depth_metrics_bf.py \
    --eval_save_dir=$EVAL_SAVE_DIR \
    --root=$BF_ROOT \
    --model_path=$MODEL_PATH

python scenerf/scripts/evaluation/agg_depth_metrics_bf.py \
    --eval_save_dir=$EVAL_SAVE_DIR \
    --root=$BF_ROOT


# Novel views synthesis on Bundlefusion
python scenerf/scripts/evaluation/render_colors_bf.py \
    --eval_save_dir=$EVAL_SAVE_DIR \
    --root=$BF_ROOT \
    --model_path=$MODEL_PATH

python scenerf/scripts/evaluation/eval_color_bf.py --eval_save_dir=$EVAL_SAVE_DIR

# Scene reconstruction on Bundlefusion
python scenerf/scripts/reconstruction/generate_novel_depths_bf.py \
    --recon_save_dir=$RECON_SAVE_DIR \
    --root=$BF_ROOT \
    --model_path=$MODEL_PATH \
    --angle=30 --step=0.2 --max_distance=2.1

python scenerf/scripts/reconstruction/depth2tsdf_bf.py \
    --recon_save_dir=$RECON_SAVE_DIR \
    --root=$BF_ROOT \
    --angle=30 --step=0.2 --max_distance=2.1

python scenerf/scripts/reconstruction/generate_sc_gt_bf.py \
    --recon_save_dir=$RECON_SAVE_DIR \
    --root=$BF_ROOT

python scenerf/scripts/evaluation/eval_sc_bf.py \
    --recon_save_dir=$RECON_SAVE_DIR \
    --root=$BF_ROOT