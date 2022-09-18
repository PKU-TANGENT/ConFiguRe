#!/bin/bash
# export CUDA_VISIBLE_DEVICES=$1
export CUDA_VISIBLE_DEVICES=4,5,6,7
python main.py +model_args=$2