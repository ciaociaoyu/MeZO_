#!/bin/bash

# Use Slurm job name as output directory (fallback if not under Slurm)
JOB_NAME=${SLURM_JOB_NAME:-manual_run}
OUT_DIR=$(pwd)/${JOB_NAME}

mkdir -p ${OUT_DIR}

TASK=${TASK:-SST-2}
K=${K:-16}
SEED=${SEED:-42}
DATA_SEED=${DATA_SEED:-$SEED}
DATASET_MODE=${DATASET_MODE:-auto}
FULL_DEV_RATIO=${FULL_DEV_RATIO:-0.1}
BS=${BS:-64}
LR=${LR:-1e-6}
EPS=${EPS:-1e-3}
WD=${WD:-0}
OPT=${OPT:-sgd}
ZERO_ORDER_USE_TRAINER_OPTIM=${ZERO_ORDER_USE_TRAINER_OPTIM:-False}
EFFICIENT_ZERO_ORDER=${EFFICIENT_ZERO_ORDER:-True}
# This is true seting
#STEP=${STEP:-100000}
#EVAL_STEP=${EVAL_STEP:-10000}
# This is for testing
STEP=${STEP:-50000}
EVAL_STEP=${EVAL_STEP:-5000}
MODEL=${MODEL:-roberta-large}
USE_H=${USE_H:-"True"}
USE_C=${USE_C:-"False"}
DATALOADER_SHUFFLE=${DATALOADER_SHUFFLE:-"False"}

LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

echo "TASK: $TASK"
echo "K: $K"
echo "Seed: $SEED"
echo "Data seed: $DATA_SEED"
echo "Dataset mode: $DATASET_MODE"
echo "Full dev ratio: $FULL_DEV_RATIO"
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "Step: $STEP; Eval step: $EVAL_STEP"
echo "Optimizer: $OPT"
echo "Use trainer optimizer for ZO: $ZERO_ORDER_USE_TRAINER_OPTIM"
echo "Efficient zero order: $EFFICIENT_ZERO_ORDER"
echo "Using adaptive h: $USE_H"
echo "Dataloader shuffle: $DATALOADER_SHUFFLE"

GR_TAG=seed$SEED-bs$BS-lr$LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP
EXTRA_TAG=${EXTRA_TAG:-ft}
TAG=${TAG:-k${K}-${MODEL}-mezo-${EXTRA_TAG}}
echo "Grid search tag: $GR_TAG"
echo "Tag: $TAG"

TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K DATASET_MODE=$DATASET_MODE DATA_SEED=$DATA_SEED FULL_DEV_RATIO=$FULL_DEV_RATIO \
    bash run_fewshot.sh --per_device_train_batch_size $BS --learning_rate $LR --eval_steps $EVAL_STEP --weight_decay $WD --zero_order_eps $EPS --use_adaptive_h $USE_H --use_c_scale $USE_C \
    --zero_order_optim --lr_scheduler_type "constant" --optimizer "$OPT" --zero_order_use_trainer_optim "$ZERO_ORDER_USE_TRAINER_OPTIM" --efficient_zero_order "$EFFICIENT_ZERO_ORDER" \
    $@
