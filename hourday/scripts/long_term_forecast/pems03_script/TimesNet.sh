# export CUDA_VISIBLE_DEVICES=7

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../data/ \
  --data_path pems03_all_common_flow.csv \
  --model_id traffic_168_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 9800 \
  --label_len 48 \
  --pred_len 9800 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 151 \
  --dec_in 151 \
  --c_out 151 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 4 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 