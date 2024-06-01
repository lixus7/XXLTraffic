export CUDA_VISIBLE_DEVICES=2

model_name=Crossformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_168_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --d_model 128 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --top_k 5 \
  --des 'Exp' \
  --n_heads 2 \
  --batch_size 2 \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'