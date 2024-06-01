# export CUDA_VISIBLE_DEVICES=1

model_name=LightTS

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../data/ \
  --data_path pems03_all_common_flow.csv \
  --model_id pems03_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 21800 \
  --label_len 48 \
  --pred_len 21800 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 151 \
  --dec_in 151 \
  --c_out 151 \
  --des 'Exp' \
  --itr 1