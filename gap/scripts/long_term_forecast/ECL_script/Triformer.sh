export CUDA_VISIBLE_DEVICES=4

model_name=Triformer

python -u Triformer_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_168_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --pred_len 192 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 32 \
  --patch_sizes 7 4 3 2\
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --batch_size 8 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'