#!/bin/bash
# export CUDA_VISIBLE_DEVICES=$1
export CUDA_VISIBLE_DEVICES=0
# for specific tasks, change `model_args` to respective config name in configs/model_args
python main.py +model_args=CRF