export CUDA_VISIBLE_DEVICES=3

model_name=CARD

  
python -u CARD_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --e_layers 2 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.3\
  --fc_dropout 0.3\
  --head_dropout 0\
  --patch_len 16\
  --stride 8\
  --batch_size 8 \
  --learning_rate 0.0001 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'