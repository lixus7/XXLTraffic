export CUDA_VISIBLE_DEVICES=4

model_name=Triformer

python -u Triformer_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_384_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 384 \
  --pred_len 192 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --patch_sizes 8 4 3 2 2\
  --d_model 32 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 8 \
  --train_epoch 200 \
  --patience 5 