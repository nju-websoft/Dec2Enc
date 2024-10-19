export MODEL_PATH='Qwen/Qwen2.5-0.5B'
export lr=1e-5
export train_micro_batch_size_per_gpu=8
export port=20013
export init=False
export epoch_num=12
#Train
deepspeed --master_port ${port} run_clean.py --train_micro_batch_size_per_gpu ${train_micro_batch_size_per_gpu} \
 --lr ${lr} --model_name ${MODEL_PATH} --epoch_num ${epoch_num} --init ${init} --only_eval ${init}


export init=True
# Evaluation
deepspeed --master_port ${port} run_clean.py --train_micro_batch_size_per_gpu ${train_micro_batch_size_per_gpu} \
 --lr ${lr} --model_name ${MODEL_PATH} --epoch_num ${epoch_num} --init ${init} --only_eval ${init}

