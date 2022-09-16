#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
python main.py \
    --train_data_path "./Data/train.json"\
    --valid_data_path "./Data/valid.json"\
    --test_data_path "./Data/test.json"\
    --task "Extraction" \
    --batch_size 16 \
    --arch_name "bert_seed2020" \
    --num_epochs 30 \
    # --crf True
