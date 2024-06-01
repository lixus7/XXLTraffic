#!/bin/bash

#PBS -q gpuvolta
#PBS -l walltime=47:59:00
#PBS -l ngpus=4
#PBS -l ncpus=48
#PBS -l mem=384GB
#PBS -l jobfs=400GB
#PBS -P wn86
#PBS -l storage=scratch/hn98+gdata/hn98
#PBS -M du.yin@unsw.edu.au
#PBS -m b
#PBS -m e
# module load cuda/10.1
# module load cudnn/7.6.5-cuda10.1
source ~/.bashrc
ca timesnet
 
nvidia-smi

cd ~/hourday

export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=TimesNet

python -u run.py \
  --use_multi_gpu \
  --train_seed 2024 \
  --samle_rate 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../data/ \
  --data_path pems10_h.csv \
  --model_id pems10_h_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 107 \
  --dec_in 107 \
  --c_out 107 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 4 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1   >> timesnet_pems10_h_in96_out96_trseed2024.log 2>&1


python -u run.py \
  --use_multi_gpu \
  --train_seed 2024 \
  --samle_rate 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../data/ \
  --data_path pems10_h.csv \
  --model_id pems10_h_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 107 \
  --dec_in 107 \
  --c_out 107 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 4 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1  >> timesnet_pems10_h_in96_out192_trseed2024.log 2>&1


python -u run.py \
  --use_multi_gpu \
  --train_seed 2024 \
  --samle_rate 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../data/ \
  --data_path pems10_h.csv \
  --model_id pems10_h_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 107 \
  --dec_in 107 \
  --c_out 107 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 4 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 >> timesnet_pems10_h_in96_out336_trseed2024.log 2>&1


python -u run.py \
  --use_multi_gpu \
  --train_seed 2024 \
  --samle_rate 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../data/ \
  --data_path pems10_h.csv \
  --model_id pems10_h_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 107 \
  --dec_in 107 \
  --c_out 107 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 4 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 >> timesnet_pems10_h_in96_out720_trseed2024.log 2>&1



