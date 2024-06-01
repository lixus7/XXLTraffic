# export CUDA_VISIBLE_DEVICES=7

model_name=DLinear

python -u run.py \
  --train_seed 2024 \
  --samle_rate 0.1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../data/ \
  --data_path pems03_all_common_flow.csv \
  --model_id pems03_all_96_96 \
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
  --enc_in 151 \
  --dec_in 151 \
  --c_out 151 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'


python -u run.py \
  --train_seed 2024 \
  --samle_rate 0.1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../data/ \
  --data_path pems03_all_common_flow.csv \
  --model_id pems03_all_96_192 \
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
  --enc_in 151 \
  --dec_in 151 \
  --c_out 151 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'

python -u run.py \
  --train_seed 2024 \
  --samle_rate 0.1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../data/ \
  --data_path pems03_all_common_flow.csv \
  --model_id pems03_all_96_336 \
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
  --enc_in 151 \
  --dec_in 151 \
  --c_out 151 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'

python -u run.py \
  --train_seed 2024 \
  --samle_rate 0.1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../data/ \
  --data_path pems03_all_common_flow.csv \
  --model_id pems03_all_96_720 \
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
  --enc_in 151 \
  --dec_in 151 \
  --c_out 151 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'
