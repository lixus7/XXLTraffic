#!/bin/bash

#PBS -q gpuvolta
#PBS -l walltime=47:59:00
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=384GB
#PBS -l jobfs=400GB
#PBS -P po67
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

model_name=FEDformer

python -u run.py \
  --train_seed 2024 \
  --gap_day 730 \
  --samle_rate 0.1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../data/pems/ \
  --data_path pems04_all_common_flow.csv \
  --model_id pems04_all_96_96 \
  --model $model_name \
  --data custom \
  --target '' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 822 \
  --dec_in 822 \
  --c_out 822 \
  --d_model 512 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3' >> fedformer_pems04_gap2_in96_out96_srate01_trseed2024.log 2>&1


python -u run.py \
  --train_seed 2024 \
  --gap_day 730 \
  --samle_rate 0.1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../data/pems/ \
  --data_path pems04_all_common_flow.csv \
  --model_id pems04_all_96_192 \
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
  --enc_in 822 \
  --dec_in 822 \
  --c_out 822 \
  --d_model 512 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'  >> fedformer_pems04_gap2_in96_out192_srate01_trseed2024.log 2>&1

python -u run.py \
  --train_seed 2024 \
  --gap_day 730 \
  --samle_rate 0.1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../data/pems/ \
  --data_path pems04_all_common_flow.csv \
  --model_id pems04_all_96_336 \
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
  --enc_in 822 \
  --dec_in 822 \
  --c_out 822 \
  --d_model 512 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'  >> fedformer_pems04_gap2_in96_out336_srate01_trseed2024.log 2>&1

python -u run.py \
  --train_seed 2024 \
  --gap_day 730 \
  --samle_rate 0.1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../data/pems/ \
  --data_path pems04_all_common_flow.csv \
  --model_id pems04_all_96_720 \
  --model $model_name \
  --data custom \
  --target '' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 822 \
  --dec_in 822 \
  --c_out 822 \
  --d_model 512 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3' >> fedformer_pems04_gap2_in96_out720_srate01_trseed2024.log 2>&1

