#!/bin/bash
python3 train_timm.py --train-ds-path /media/minh/01D8D2B8C5178CE0/datasets/dogs-vs-cats/images/train \
                      --val-ds-path /media/minh/01D8D2B8C5178CE0/datasets/dogs-vs-cats/images/val \
                      --input_size 224 \
                      --normalize_mean 0.485 0.456 0.406 \
                      --normalize_std 0.229 0.224 0.225 \
                      --aug_strength weak \
                      --batch_size 8 \
                      --ema \
                      --optimizer adamw \
                      --learning_rate 0.0001 \
                      --arch convnext_small \
                      --dropout 0.2 \
                      --use_pretrain \
                      --freeze_bottom \
                      --n_block_to_train 1 \
                      --smoothing 0.0 \
                      --epochs 50 \
                      --seed 42 \
                      --ckpt-path weights/ \
                      --log-path logs/ \
                      --device 0 \
                      --n-worker 4
