#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
cd /fs-computility/video/shared/linhan/code/Qwen2.5-VL/qwen-vl-finetune
# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
# llm=/fs-computility/video/shared/linhan/models/Qwen2-VL-7B  # Using HuggingFace model ID
llm=/tos-bjml-video/linhan/models/Qwen2-VL-2B/

# Training hyperparameters
lr=2e-7
batch_size=4
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
# datasets=llava_image_tune,videochatgpt
# datasets=llava_image_tune_random6,videochatgpt_random6
# datasets=llava_image_tune
# Output configuration
# run_name="qwen2vl-llava_videochatgpt_nonlp_random0.06"
# run_name="qwen2vl_llava"
# run_name="test_nonlp"
output_dir=./output/${run_name}
mkdir -p ${output_dir}
log_dir=${output_dir}_$(date +%Y-%m-%d-%H-%M-%S).log

export WANDB_MODE=offline
export WANDB_API_KEY="e619ca3afd29c4b7fc11e2d597897401d617f568"
# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --data_flatten True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=$MLP_WORKER_GPU \
        --nnodes=$MLP_WORKER_NUM \
        --node_rank=$MLP_ROLE_INDEX \
        --master_addr=$MLP_WORKER_0_HOST \
        --master_port=$MLP_WORKER_0_PORT \
         ${entry_file} ${args} > ${log_dir} 2>&1