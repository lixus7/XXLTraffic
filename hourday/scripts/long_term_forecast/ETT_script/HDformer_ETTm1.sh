export CUDA_VISIBLE_DEVICES=6

model_name=HDformer

python -u my_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_384_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 384 \
  --label_len 48 \
  --pred_len 96 \
  --cycle_len 96 \
  --short_period_len 8 \
  --kernel_size 3 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 6 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --lradj 'type3' \
  --itr 1 \
  --train_epoch 200 \
  --patience 5