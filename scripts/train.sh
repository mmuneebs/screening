#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python trainer.py --epochs 120 --lrate 0.1 --ext aps --datadir /home/muneeb/ml/data/ \
--batch 20 --labelsmooth 0.0 --padding valid --dropout 0.3 --decaysteps 80 --traindir /home/muneeb/ml/tsa/ckpt
