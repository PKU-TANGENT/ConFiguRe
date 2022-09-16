#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
python -m debugpy --listen 127.0.0.1:9999 --wait-for-client main.py \
    --train_data_path "./Data/train.json"\
    --valid_data_path "./Data/valid.json"\
    --test_data_path "./Data/test.json"\
    --task "CRF" \
    --batch_size 16 \
    --arch_name "CRF" \
    --num_epochs 2 \
    --crf True
