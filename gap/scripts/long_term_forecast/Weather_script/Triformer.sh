export CUDA_VISIBLE_DEVICES=3

model_name=Triformer

python -u Triformer_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_432_3 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 432 \
  --pred_len 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --patch_sizes 6 6 4 3\
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --batch_size 8 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'

python -u Triformer_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_432_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 432 \
  --pred_len 24 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --patch_sizes 6 6 4 3\
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --batch_size 8 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'
  
python -u Triformer_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_432_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 432 \
  --pred_len 96 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --patch_sizes 6 6 4 3\
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --batch_size 8 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'
  
python -u Triformer_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_432_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 432 \
  --pred_len 192 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --patch_sizes 6 6 4 3\
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --batch_size 8 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'