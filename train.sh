#!/bin/bash

nohup python train.py \
    --batch_size 4 \
    --max_epochs 500 \
    --save_checkpoint lightning_logs/rx3d_1e-3_500e \
    > train.log &
