#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -m debugpy --listen 127.0.0.1:9999 --wait-for-client main.py \
    +model_args=CRF