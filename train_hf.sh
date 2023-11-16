#!/bin/bash
python3 train_hf.py --train-ds-path /media/minh/01D8D2B8C5178CE0/datasets/dogs-vs-cats/images/train \
                    --val-ds-path /media/minh/01D8D2B8C5178CE0/datasets/dogs-vs-cats/images/val \
                    --input_size 224 \
                    --normalize_mean 0.485 0.456 0.406 \
                    --normalize_std 0.229 0.224 0.225 \
                    --aug_strength weak \
                    --batch_size 8 \
                    --optimizer adamw \
                    --learning_rate 0.0001 \
                    --model_name_or_path efficientnet_b1 \
                    --use_pretrain \
                    --epochs 50 \
                    --seed 42 \
                    --n-worker 4
