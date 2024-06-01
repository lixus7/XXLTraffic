export CUDA_VISIBLE_DEVICES=3

model_name=SCNN

python -u my_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_168_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 192 \
  --cycle_len 24 \
  --short_period_len 8 \
  --kernel_size 2 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 8 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --lradj 'type3' \
  --itr 1 \
  --train_epoch 200 \
  --patience 5