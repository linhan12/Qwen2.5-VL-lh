#!/bin/bash
set -ex
# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
ROOT_DIR="/mnt/shared-storage-user/linhan"
REPO_NAME="Qwen2.5-VL-lh"

cd ${ROOT_DIR}/${REPO_NAME}/qwen-vl-finetune
# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
# llm=/fs-computility/video/shared/linhan/models/Qwen2-VL-7B  # Using HuggingFace model ID
llm=/mnt/shared-storage-user/linhan/oss_models/Qwen2-VL-7B

# Training hyperparameters
lr=2e-7
batch_size=1
grad_accum_steps=1
epoch=1
# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
# datasets=llava_image_tune,videochatgpt
# datasets=llava_image_tune_random6,videochatgpt_random6

# datasets=llava_image_tune_lessmm1678080
# Output configuration
# run_name="qwen2vl-llava_videochatgpt_nonlp_random0.06"
# run_name="qwen2vl_llava"
run_name="test_nonlp"

batch_size=16
datasets=llava_image_tune_sel_groupaug_block2
run_name="qwen2vl_llava_sel_groupaug_block2"

batch_size=1
datasets=llava_image_tune_sel_groupaug_block2_less671230_v2
run_name="qwen2vl_llava_sel_groupaug_block2_less671230_v2"

# batch_size=1
# datasets=llava_image_tune_sel_groupaug_block2
# run_name="qwen2vl_llava_sel_groupaug_block2"

batch_size=1
datasets=llava_image_tune_sel_groupaug_block1
run_name="qwen2vl_llava_sel_groupaug_block1"

batch_size=16
datasets=llava_image_tune_sel_groupaug_block2_less201369
run_name="qwen2vl_llava_sel_groupaug_block2_less201369_epoch${epoch}_bs${batch_size}"

batch_size=16
datasets=llava_image_tune_clip006
run_name="qwen2vl_llava_clip006_epoch${epoch}_bs${batch_size}"

batch_size=16
datasets=llava_image_tune_coincide006
run_name="llava_image_tune_coincide006_epoch${epoch}_bs${batch_size}"

batch_size=16
datasets=llava_image_tune_random6
run_name="llava_image_tune_random6_epoch${epoch}_bs${batch_size}"

batch_size=8
epoch=2
datasets=llava_image_tune_sel_aug_block3_random201369
run_name="qwen2vl_llava_sel_aug_block3_random201369_epoch${epoch}_bs${batch_size}"

# batch_size=8
# epoch=2
# datasets=llava_image_tune_sel_groupaug_block2_random201369
# run_name="qwen2vl_llava_sel_groupaug_block2_random201369_epoch${epoch}_bs${batch_size}"

batch_size=8
epoch=2
datasets=llava_image_tune_sel_aug_block1_random201369
run_name="qwen2vl_llava_sel_aug_block1_random201369_epoch${epoch}_bs${batch_size}"

batch_size=8
epoch=2
datasets=abl_random_single_aug_group_aug_random201369
run_name="qwen2vl_llava_abl_random_single_aug_group_aug_random201369_epoch${epoch}_bs${batch_size}"

# batch_size=16
# batch_size=8
# epoch=2
# datasets=abl_coincide_single_aug_group_aug_random201369
# run_name="qwen2vl_llava_abl_coincide_single_aug_group_aug_random201369_epoch${epoch}_bs${batch_size}"

# batch_size=16
# epoch=2
# datasets=abl_coincide_less_random_random201369
# run_name="qwen2vl_llava_abl_coincide_less_random_random201369"

# batch_size=16
# datasets=llava_image_tune,nlp_tune
# run_name="qwen2vl_llava_fulldata_nlp"

# batch_size=16
# batch_size=8
# epoch=2
# datasets=abl_less_single_aug_group_aug_random201369
# run_name="qwen2vl_llava_abl_less_single_aug_group_aug_random201369_epoch${epoch}_bs${batch_size}"

# batch_size=8
# epoch=2
# datasets=llava_image_tune_sel_aug_block1_less201369
# run_name="qwen2vl_llava_sel_aug_block1_less201369_epoch${epoch}_bs${batch_size}"


# batch_size=16
# batch_size=8
# epoch=2
# datasets=abl_less_single_aug_group_aug_random_single_aug_group_aug_random201369
# run_name="qwen2vl_llava_abl_less_single_aug_group_aug_random_single_aug_group_aug_random20136_epoch${epoch}_bs${batch_size}"

# batch_size=16
# batch_size=8
# epoch=2
# datasets=abl_coincide_single_aug_group_aug_random_single_aug_group_aug_random201369
# run_name="qwen2vl_llava_abl_coincide_single_aug_group_aug_random_single_aug_group_aug_random201369_epoch${epoch}_bs${batch_size}"

# batch_size=16
# batch_size=8
# epoch=2
# datasets=abl_coincide_single_aug_group_aug_less_single_aug_group_aug_random201369
# run_name="qwen2vl_llava_abl_coincide_single_aug_group_aug_less_single_aug_group_aug_random201369_epoch${epoch}_bs${batch_size}"

# batch_size=16
# batch_size=8
# epoch=2
# datasets=abl_coincide_single_aug_less_single_aug_random_single_aug_random201369
# run_name="qwen2vl_llava_abl_coincide_single_aug_less_single_aug_random_single_aug_random201369_epoch${epoch}_bs${batch_size}"


# batch_size=8
# epoch=2
# datasets=abl_coincide_group_aug_less_groupaug_random_group_aug_random201369
# run_name="qwen2vl_llava_abl_coincide_group_aug_less_groupaug_random_group_aug_random201369_epoch${epoch}_bs${batch_size}"

# batch_size=1
# datasets=llava_image_tune_sel_groupaug_block1
# run_name="qwen2vl_llava_sel_groupaug_block1"

# batch_size=16
# datasets=llava_image_tune_lessmm1678080
# run_name="qwen2vl_llava_lessmm1678080"

# batch_size=16
# datasets=llava_image_tune_random1678080
# run_name="qwen2vl_llava_random1678080"

# batch_size=16
# datasets=llava_image_tune_random0.2
# run_name="qwen2vl_llava_random0.2"

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
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 1 \
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
    # --save_strategy "step" \
    # --save_steps 1000 \
    # --save_total_limit 1 \
# Launch training
# torchrun --nproc_per_node=$MLP_WORKER_GPU \
#         --nnodes=$MLP_WORKER_NUM \
#         --node_rank=$MLP_ROLE_INDEX \
#         --master_addr=$MLP_WORKER_0_HOST \
#         --master_port=$MLP_WORKER_0_PORT \
torchrun --nproc_per_node=8 \
         --nnodes=1 \
         ${entry_file} ${args} > ${log_dir} 2>&1 &
