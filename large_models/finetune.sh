MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
TASK=${TASK:-SST-2}

EPOCH=${EPOCH:-5}
BS=${BS:-8}
LR=${LR:-1e-5}
SEED=${SEED:-0}
K=${K:-16}
DATASET_MODE=${DATASET_MODE:-auto}
DATA_SEED=${DATA_SEED:-$SEED}
FULL_DEV_RATIO=${FULL_DEV_RATIO:-0.1}
TRAIN=${TRAIN:-}
DEV=${DEV:-}
EVAL=${EVAL:-}
# MeZO epsilon (finite-diff step size)
ZO_EPS=${ZO_EPS:-1e-3}

MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi
TAG=$MODE-$EPOCH-$BS-$LR-$SEED

TASK_ARGS=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Previously used DEV=100; for full-data training we do not split dev by default.
        ;;
    Copa) # It has <1000 training examples. Previously used DEV=100; for full-data training we do not split dev by default.
        TASK_ARGS="--train_as_classification False"
        ;;
    MultiRC) # Can only fit real bsz = 2 on 80G A100
        GA=$(expr $BS / 2)
        BS=2
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA"
        ;;
    ReCoRD) # Can only fit real bsz = 2 on 80G A100
        GA=$(expr $BS / 2)
        BS=2
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
        ;;
    DROP) # Can only fit real bsz = 1 on 80G A100
        GA=$(expr $BS / 1)
        BS=1
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

echo $TAG
echo "EPOCH: $EPOCH"
echo "BS: $BS"
echo "LR: $LR"
echo "SEED: $SEED"
echo "DATASET_MODE: $DATASET_MODE"
echo "K: $K"
echo "DATA_SEED: $DATA_SEED"
echo "FULL_DEV_RATIO: $FULL_DEV_RATIO"
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
    --trainer regular --fp16 \
    --zo_eps $ZO_EPS \
    --learning_rate $LR --num_train_epochs $EPOCH --per_device_train_batch_size $BS \
    --load_best_model_at_end --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1 \
    --train_as_classification \
    "${LEGACY_ARGS[@]}" \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"
