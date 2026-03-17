MODEL=${MODEL:-facebook/opt-2.7b}
TASK=${TASK:-MNLI}
MODEL_NAME="${MODEL##*/}"

MODE=${MODE:-ft}
LR=${LR:-1e-6}
EPS=${EPS:-1e-3}
EPOCH=${EPOCH:-1}
BS=${BS:-1}
GA=${GA:-8}
SEED=${SEED:-0}

# Full-data defaults
TRAIN=${TRAIN:--1}
DEV=${DEV:-0}
EVAL=${EVAL:--1}

EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi

TAG=mezo-opt2p7b-mnli-$MODE-ep$EPOCH-bs$BS-ga$GA-lr$LR-eps$EPS-seed$SEED

echo $TAG
echo "MODEL: $MODEL"
echo "TASK: $TASK (full train: $TRAIN, eval: $EVAL, dev split from train: $DEV)"
echo "MODE: $MODE"
echo "EPOCH: $EPOCH"
echo "BS/GA: $BS/$GA"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "Extra args: $EXTRA_ARGS"

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG \
    --tag $TAG \
    --train_set_seed $SEED \
    --num_train $TRAIN \
    --num_dev $DEV \
    --num_eval $EVAL \
    --trainer zo \
    --load_float16 \
    --train_as_classification \
    --learning_rate $LR \
    --zo_eps $EPS \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BS \
    --gradient_accumulation_steps $GA \
    --lr_scheduler_type constant \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --logging_steps 10 \
    $EXTRA_ARGS \
    "$@"
