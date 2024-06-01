export CUDA_VISIBLE_DEVICES=1

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_168_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --itr 1 \
  --train_epoch 200 \
  --patience 5 \
  --batch_size 1 \
  --lradj 'type3'