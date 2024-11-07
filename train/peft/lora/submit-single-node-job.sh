#!/bin/bash

export MODEL_NAME=${MODEL_NAME:-Meta-Llama-3.1-8B-Instruct}
export GPUS_PER_NODE=${GPUS_PER_NODE:-1}
export EPOCHS=${EPOCHS:-2}
export TRAIN_BATCH_SIZE_PER_DEVICE=${TRAIN_BATCH_SIZE_PER_DEVICE:-2}
export DATASET_TEMPLATE=${DATASET_TEMPLATE:-llama3}
export REPORT_TO=${REPORT_TO:-tensorboard}
export LOGGING_DIR=${LOGGING_DIR:-/app/output/logs}

envsubst '$MODEL_NAME $GPUS_PER_NODE $EPOCHS $$TRAIN_BATCH_SIZE_PER_DEVICE $DATASET_TEMPLATE $REPORT_TO $LOGGING_DIR' < lora-example-job.tpl.yaml | kubectl replace --force -f -
