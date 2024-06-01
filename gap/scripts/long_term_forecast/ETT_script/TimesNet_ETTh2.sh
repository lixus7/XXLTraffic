export CUDA_VISIBLE_DEVICES=5

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_168_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1