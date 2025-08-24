#!/bin/bash

#$ -P CRUISE
#$ -N 2xxltra2
#$ -j y
#$ -m ea
#$ -M du.yin@unsw.edu.au
#$ -e /srv/scratch/CRUISE/Du/exlts/$JOB_ID_$JOB_NAME.err
#$ -o /srv/scratch/CRUISE/Du/exlts/$JOB_ID_$JOB_NAME.out
#$ -cwd
#$ -l walltime=100:00:00
#$ -l mem=480G
#$ -l jobfs=200G
#$ -l tmpfree=160G
#$ -l ngpus=1
#$ -pe smp 2
#$ -l gpu_model=L40S


__conda_setup="$('/srv/scratch/CRUISE/Du/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/srv/scratch/CRUISE/Du/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/srv/scratch/CRUISE/Du/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/srv/scratch/CRUISE/Du/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate torch2
 
cd /srv/scratch/CRUISE/Du/exlts/ddd2

model_name=agcrn

python -u run.py \
  --gap_day 730 \
  --batch_size  8 \
  --train_seed 2024 \
  --samle_rate 0.1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --adj_path ../../data/W_pems04.csv \
  --root_path ../../data/pems/ \
  --data_path pems04_all_common_flow.csv \
  --model_id pems04_all_96_96 \
  --model $model_name \
  --data custom \
  --target '' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 822 \
  --dec_in 822 \
  --c_out 822 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3' >> agcrn_pems04_gap2_in96_out96_srate01_trseed2024.log 2>&1


python -u run.py \
  --gap_day 730 \
  --batch_size  8 \
  --train_seed 2024 \
  --samle_rate 0.1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --adj_path ../../data/W_pems04.csv \
  --root_path ../../data/pems/ \
  --data_path pems04_all_common_flow.csv \
  --model_id pems04_all_96_192 \
  --model $model_name \
  --data custom \
  --target '' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 822 \
  --dec_in 822 \
  --c_out 822 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'  >> agcrn_pems04_gap2_in96_out192_srate01_trseed2024.log 2>&1

python -u run.py \
  --gap_day 730 \
  --batch_size  8 \
  --train_seed 2024 \
  --samle_rate 0.1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --adj_path ../../data/W_pems04.csv \
  --root_path ../../data/pems/ \
  --data_path pems04_all_common_flow.csv \
  --model_id pems04_all_96_336 \
  --model $model_name \
  --data custom \
  --target '' \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 822 \
  --dec_in 822 \
  --c_out 822 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'  >> agcrn_pems04_gap2_in96_out336_srate01_trseed2024.log 2>&1


