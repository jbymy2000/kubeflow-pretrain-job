#!/bin/bash
set -x

echo "===> submit job for JOBNAME: $JOB_NAME"

WORK_DIR='/workspace/LLaMA-Factory'
MODEL_PATH=/data/pretrain/model/deepseek-llm-7b-base
OUTPUT_DIR=/data/pretrain/model/deepseek-llm-7b-base/pretrained_model_v2

# Distributed training settings
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=$PET_NNODES
NODE_RANK=$PET_NODE_RANK

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# NCCL Settings
export NCCL_IB_HCA='mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_11'
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_TIMEOUT_MS=60000
export NCCL_LAUNCH_TIMEOUT=60000
export NCCL_COMM_ID_TIMEOUT=60000
export NCCL_SOCKET_TIMEOUT=60000

# Training Arguments
TRAINING_ARGS="
    --stage pt \
    --do_train \
    --model_name_or_path $MODEL_PATH \
    --dataset wiki_demo \
    --finetuning_type full \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len ${SEQ_LENGTH:-4096} \
    --preprocessing_num_workers 160 \
    --per_device_train_batch_size ${MICRO_BATCH_SIZE:-2} \
    --per_device_eval_batch_size ${MICRO_BATCH_SIZE:-2} \
    --gradient_accumulation_steps ${GRAD_ACC_STEPS:-4} \
    --lr_scheduler_type cosine \
    --logging_steps ${LOG_STEPS:-1} \
    --warmup_steps ${WARMUP_STEPS:-1000} \
    --save_steps ${SAVE_INTERVAL:-2000} \
    --eval_steps ${EVAL_INTERVAL:-1000} \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate ${LR:-5e-5} \
    --num_train_epochs ${EPOCHS:-2} \
    --val_size 10 \
    --plot_loss \
    --bf16 true \
    --warmup_ratio 0.02 \
    --ddp_timeout 300000000 \
    --save_on_each_node False \
    --flash_attn auto \
    --gradient_checkpointing True \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --save_total_limit 10 \
    --max_steps ${TRAIN_ITERS:-100} \
    --streaming true \
    --deepspeed examples/deepspeed/ds_z3_config.json
"

# Clone and install LLaMA Factory
cd /workspace
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install flash-attn --no-build-isolation
pip install deepspeed

# Launch training
echo "===> launch training job ..."
torchrun $DISTRIBUTED_ARGS src/train.py $TRAINING_ARGS

echo "===> job finished."
