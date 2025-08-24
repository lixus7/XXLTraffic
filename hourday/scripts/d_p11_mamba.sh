#!/bin/bash

#PBS -q gpuvolta
#PBS -l walltime=47:59:00
#PBS -l ngpus=4
#PBS -l ncpus=48
#PBS -l mem=382GB
#PBS -l jobfs=400GB
#PBS -P wn86
#PBS -l storage=scratch/hn98+gdata/hn98
#PBS -M du.yin@unsw.edu.au
#PBS -m b
#PBS -m e
# module load cuda/10.1
# module load cudnn/7.6.5-cuda10.1
source ~/.bashrc
ca mamba
 
nvidia-smi

cd /g/data/hn98/du/exlts/hourdayweek

# export CUDA_VISIBLE_DEVICES=7

model_name=Mamba

CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --train_seed 2024 \
  --samle_rate 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../data/pems/ \
  --data_path pems11_d.csv \
  --model_id pems11_d_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --expand 2 \
  --enc_in 521 \
  --c_out 521 \
  --d_model 128 \
  --d_ff 16 \
  --d_conv 4 \
  --des 'Exp' \
  --itr 1   >> mamba_pems11_d_in96_out96_trseed2024.log 2>&1 & \


CUDA_VISIBLE_DEVICES=1 python -u run.py \
  --train_seed 2024 \
  --samle_rate 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../data/pems/ \
  --data_path pems11_d.csv \
  --model_id pems11_d_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --expand 2 \
  --enc_in 521 \
  --c_out 521 \
  --d_model 128 \
  --d_ff 16 \
  --d_conv 4 \
  --des 'Exp' \
  --itr 1  >> mamba_pems11_d_in96_out96_trseed2024.log 2>&1 & \


CUDA_VISIBLE_DEVICES=2 python -u run.py \
  --train_seed 2024 \
  --samle_rate 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../data/pems/ \
  --data_path pems11_d.csv \
  --model_id pems11_d_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --expand 2 \
  --enc_in 521 \
  --c_out 521 \
  --d_model 128 \
  --d_ff 16 \
  --d_conv 4 \
  --des 'Exp' \
  --itr 1 >> mamba_pems11_d_in96_out96_trseed2024.log 2>&1


