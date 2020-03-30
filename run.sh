#!/bin/bash

python src/run_csqa.py --data_dir data/ --model_type roberta-ocn-inj --model_name_or_path roberta-large --task_name csqa-inj --overwrite_output_dir --cache_dir downloaded_models --max_seq_length 80 --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 1e-5 --num_train_epochs 8 --warmup_steps 150 --output_dir workspace/hykas --no_cuda
