#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
python main.py +model_args=$2 \
    +model_args.eval_only=True \
    +hydra.job_logging.handlers.file.mode=a