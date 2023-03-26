#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../

CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch example.py --local_rank=0 \
  /mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/pretrain_models/llama/7B \
  /mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/ft_52k/llama-7b-hf_train_out/checkpoint-200/tokenizer.model

