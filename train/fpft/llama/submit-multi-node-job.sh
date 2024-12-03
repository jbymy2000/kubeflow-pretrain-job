#!/bin/bash

export NNODES=${NNODES:-2}
export WORKER_NODES=$((NNODES - 1))
export MODEL_NAME=${MODEL_NAME:-deepseek-llm-7b-base}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

export IMAGE_NAME=${IMAGE_NAME:-10.5.1.249/ebtech-website/pytorch:v2.2.0-cuda12.1-cudnn8-devel}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-4}
export SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
export TRAIN_ITERS=${TRAIN_ITERS:-100}
export EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
export SEQ_LENGTH=${SEQ_LENGTH:-4096}
export LOG_STEPS=${LOG_STEPS:-1}
export WARMUP_STEPS=${WARMUP_STEPS:-1000}
export LR=${LR:-5e-5}
export EPOCHS=${EPOCHS:-2}

# Replace environment variables in template
envsubst '$MODEL_NAME $NNODES $WORKER_NODES $GPUS_PER_NODE \
          $IMAGE_NAME $MICRO_BATCH_SIZE $GRAD_ACC_STEPS \
          $SAVE_INTERVAL $TRAIN_ITERS $EVAL_INTERVAL \
          $SEQ_LENGTH $LOG_STEPS $WARMUP_STEPS $LR $EPOCHS \
        ' < example-pytorchjob.tpl.yaml | kubectl replace --force -f -
