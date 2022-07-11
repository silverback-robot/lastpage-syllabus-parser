#!/bin/bash

cd ../tools

python train_net.py \
    --dataset_name          annotations \
    --json_annotation_train ../data/annotations/train.json \
    --image_path_train      ../data/annotations/ \
    --json_annotation_val   ../data/annotations/test.json \
    --image_path_val        ../data/annotations/ \
    --num-gpus              1 \ # UPDATE BEFORE STARTING TRAINING
    --config-file           ../configs/prima/fast_rcnn_R_50_FPN_3x.yaml \
    OUTPUT_DIR  ../outputs/lastpage/fast_rcnn_R_50_FPN_3x/ \
    SOLVER.IMS_PER_BATCH 2 
