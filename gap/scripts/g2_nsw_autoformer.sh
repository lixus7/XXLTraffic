#!/bin/bash

#PBS -q gpuvolta
#PBS -l walltime=47:59:00
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l jobfs=50GB
#PBS -P hn98
#PBS -l storage=scratch/hn98+gdata/hn98
#PBS -M du.yin@unsw.edu.au
#PBS -m b
#PBS -m e
# module load cuda/10.1
# module load cudnn/7.6.5-cuda10.1
source ~/.bashrc
ca timesnet
 
nvidia-smi

cd /g/data/hn98/du/exlts/ddd2

model_name=Autoformer

python -u run.py \
  --gap_day 61 \
  --train_seed 2024 \
  --samle_rate 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../data/ \
  --data_path tfnsw.csv \
  --model_id tfnsw_all_96_96 \
  --model $model_name \
  --data custom \
  --target '' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --d_model 128 \
  --factor 3 \
  --enc_in 27 \
  --dec_in 27 \
  --c_out 27 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3' >> dlinear_tfnsw_gap2_in96_out96_srate1_trseed2024.log 2>&1


python -u run.py \
  --gap_day 61 \
  --train_seed 2024 \
  --samle_rate 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../data/ \
  --data_path tfnsw.csv \
  --model_id tfnsw_all_96_192 \
  --model $model_name \
  --data custom \
  --target '' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --d_model 128 \
  --enc_in 27 \
  --dec_in 27 \
  --c_out 27 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'  >> dlinear_tfnsw_gap2_in96_out192_srate1_trseed2024.log 2>&1

python -u run.py \
  --gap_day 61 \
  --train_seed 2024 \
  --samle_rate 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../data/ \
  --data_path tfnsw.csv \
  --model_id tfnsw_all_96_336 \
  --model $model_name \
  --data custom \
  --target '' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --d_model 128 \
  --enc_in 27 \
  --dec_in 27 \
  --c_out 27 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'  >> dlinear_tfnsw_gap2_in96_out336_srate1_trseed2024.log 2>&1

python -u run.py \
  --gap_day 61 \
  --train_seed 2024 \
  --samle_rate 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../data/ \
  --data_path tfnsw.csv \
  --model_id tfnsw_all_96_720 \
  --model $model_name \
  --data custom \
  --target '' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --d_model 128 \
  --factor 3 \
  --enc_in 27 \
  --dec_in 27 \
  --c_out 27 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3' >> dlinear_tfnsw_gap2_in96_out720_srate1_trseed2024.log 2>&1

