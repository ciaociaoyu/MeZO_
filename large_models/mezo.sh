MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
TASK=${TASK:-SST-2}

BS=${BS:-16}
LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
K=${K:-16}
DATASET_MODE=${DATASET_MODE:-auto}
DATA_SEED=${DATA_SEED:-$SEED}
FULL_DEV_RATIO=${FULL_DEV_RATIO:-0.1}
TRAIN=${TRAIN:-}
DEV=${DEV:-}
EVAL=${EVAL:-}

STEPS=${STEPS:-20000}

# 临时测试一下eval_loss输出是否正确
# emmm 好像完全可以参数传递啊
EVAL_STEPS=${EVAL_STEPS:-1000}
#EVAL_STEPS=${EVAL_STEPS:-30}

MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi
TAG=mezo-$MODE-$STEPS-$BS-$LR-$EPS-$SEED

TASK_ARGS=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

echo $TAG
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "DATASET_MODE: $DATASET_MODE"
echo "K: $K"
echo "DATA_SEED: $DATA_SEED"
echo "FULL_DEV_RATIO: $FULL_DEV_RATIO"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

LEGACY_ARGS=()
if [[ -n "$TRAIN" ]]; then
    LEGACY_ARGS+=(--num_train "$TRAIN")
fi
if [[ -n "$DEV" ]]; then
    LEGACY_ARGS+=(--num_dev "$DEV")
fi
if [[ -n "$EVAL" ]]; then
    LEGACY_ARGS+=(--num_eval "$EVAL")
fi

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --logging_steps 10 \
    --dataset_mode $DATASET_MODE --num_k $K --data_seed $DATA_SEED --full_dev_ratio $FULL_DEV_RATIO \
    --max_steps $STEPS \
    --trainer zo --load_float16 \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type "constant" \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --train_as_classification \
    "${LEGACY_ARGS[@]}" \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"
