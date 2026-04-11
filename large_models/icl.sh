MODEL=${MODEL:-facebook/opt-13b}
TASK=${TASK:-SST-2}
K=${K:-16}
DATASET_MODE=${DATASET_MODE:-auto}
DATA_SEED=${DATA_SEED:-0}

python run.py --model_name $MODEL --task_name $TASK --output_dir result/tmp --tag icl --dataset_mode $DATASET_MODE --num_k $K --data_seed $DATA_SEED --num_train 32 --num_eval 1000 --load_float16 --verbose "$@"
