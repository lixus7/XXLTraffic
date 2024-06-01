#!/bin/bash

#PBS -q gpuvolta
#PBS -l walltime=47:59:00
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=380GB
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

model_name=DLinear

python -u run.py \
  --train_seed 2024 \
  --samle_rate 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../data/ \
  --data_path pems03_h.csv \
  --model_id pems03_h_720_96 \
  --model $model_name \
  --data custom \
  --target '' \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 151 \
  --dec_in 151 \
  --c_out 151 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3' >> dlinear_pems03_h_in720_out96_trseed2024.log 2>&1



python -u run.py \
  --train_seed 2024 \
  --samle_rate 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../data/ \
  --data_path pems03_h.csv \
  --model_id pems03_h_720_192 \
  --model $model_name \
  --data custom \
  --target '' \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 151 \
  --dec_in 151 \
  --c_out 151 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3' >> dlinear_pems03_h_in720_out192_trseed2024.log 2>&1


python -u run.py \
  --train_seed 2024 \
  --samle_rate 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../data/ \
  --data_path pems03_h.csv \
  --model_id pems03_h_720_336 \
  --model $model_name \
  --data custom \
  --target '' \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 151 \
  --dec_in 151 \
  --c_out 151 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3' >> dlinear_pems03_h_in720_out336_trseed2024.log 2>&1


python -u run.py \
  --train_seed 2024 \
  --samle_rate 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../data/ \
  --data_path pems03_h.csv \
  --model_id pems03_h_720_720 \
  --model $model_name \
  --data custom \
  --target '' \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 151 \
  --dec_in 151 \
  --c_out 151 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3' >> dlinear_pems03_h_in720_out720_trseed2024.log 2>&1


