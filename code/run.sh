#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
python main.py +model_args=$2