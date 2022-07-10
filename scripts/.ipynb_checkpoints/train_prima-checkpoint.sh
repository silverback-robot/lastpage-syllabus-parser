#!/bin/bash

cd ../tools

#python convert_prima_to_coco.py \
#    --prima_datapath ../data/prima \
#    --anno_savepath ../data/prima/annotations.json 

python train_net.py \
    --dataset_name          annotations \
    --json_annotation_train /home/data/annotations/train.json \
    --image_path_train      /home/data/annotations/ \
    --json_annotation_val   /home/data/annotations/test.json \
    --image_path_val        /home/data/annotations/ \
    --num-gpus              1 \
    --config-file           /home/layout-model-training/configs/prima/fast_rcnn_R_50_FPN_3x.yaml \
    OUTPUT_DIR  /home/outputs/lastpage/fast_rcnn_R_50_FPN_3x/ \
    SOLVER.IMS_PER_BATCH 2 
