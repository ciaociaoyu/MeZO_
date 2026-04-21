#!/bin/bash

# Main settings with default values
TASK=${TASK:-"SST-2"}           # see all the options in the "cases" below
SEED=${SEED:-13}                # random seed and also data seed, by default the data split seeds are {13, 21, 42, 87, 100}
K=${K:-16}                      # choose from {16, 64, 512} by default
MODEL=${MODEL:-"roberta-large"}  # pick a RoBERTa or BERT model
TYPE=${TYPE:-"prompt"}          # fine-tuning setting, choose from "finetune" and "prompt"
TRAINER=${TRAINER:-"standard"}  # choose from "standard" and "linearhead"
TAG=${TAG:-}                    # set a tag to distinguish and aggregate runs in the log
NUM_GPU=${NUM_GPU:-1}           # by default use 1 GPU, set to 0 for CPU-only training
OPT=${OPT:-"adam"}
USE_H=${USE_H:-"True"}
USE_C=${USE_C:-"False"}
DATALOADER_SHUFFLE=${DATALOADER_SHUFFLE:-"True"}
DATA_SEED=${DATA_SEED:-$SEED}
DATASET_MODE=${DATASET_MODE:-"auto"}
DATA_ROOT=${DATA_ROOT:-"data/k-shot-1k-test"}
FULL_DEV_RATIO=${FULL_DEV_RATIO:-0.1}
STEPS=${STEPS:-1000}

# ---------------------------
# Output directory layout
# Default: result/<SLURM_JOB_NAME>/seed<SEED>/
# Override:
#   --result_root /path/to/results
#   --job_name my_job
# ---------------------------
RESULT_ROOT=${RESULT_ROOT:-result}
JOB_NAME=${JOB_NAME:-${SLURM_JOB_NAME:-}}

# Parse our own script-only args (remove them from args forwarded to python).
PY_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --result_root|--result_dir|--output_root)
            RESULT_ROOT="$2"; shift 2 ;;
        --job_name|--job)
            JOB_NAME="$2"; shift 2 ;;
        --dataset_mode)
            DATASET_MODE="$2"; shift 2 ;;
        --data_seed)
            DATA_SEED="$2"; shift 2 ;;
        --data_root|--data_dir_root)
            DATA_ROOT="$2"; shift 2 ;;
        --full_dev_ratio)
            FULL_DEV_RATIO="$2"; shift 2 ;;
        *)
            PY_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$JOB_NAME" ]]; then
    JOB_NAME="local"
fi

JOB_NAME_SAFE=$(echo "$JOB_NAME" | tr '/ ' '__')
OUTPUT_DIR="$RESULT_ROOT/$JOB_NAME_SAFE/seed$SEED"

echo "Result root: $RESULT_ROOT"
echo "Job name: $JOB_NAME_SAFE"
echo "Output dir: $OUTPUT_DIR"
echo "Dataset mode: $DATASET_MODE"
echo "Data root: $DATA_ROOT"
echo "Full dev ratio: $FULL_DEV_RATIO"



# Convert DATALOADER_SHUFFLE (string) to the CLI flag expected by HfArgumentParser
# In run.py, `dataloader_shuffle` is a bool field with default True, so disabling uses `--no_dataloader_shuffle`.
SHUFFLE_FLAG=""
case "$DATALOADER_SHUFFLE" in
  False|false|0|NO|no|N|n) SHUFFLE_FLAG="--no_dataloader_shuffle" ;;
esac

echo "GPU数量: $NUM_GPU"

TASK_EXTRA=""
case $TASK in
    BoolQ|boolq)
        TEMPLATE=*cls**sent_0*_Question:*+sent_1*_Answer:*mask*.*sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        TASK_EXTRA="--max_seq_length 512 --first_sent_limit 384 --other_sent_limit 64"
        ;;
    SST-2|sst-2)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{'0':'terrible','1':'great'}"
        ;;
    SST-5|sst-5)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 20 --double_demo"
        ;;
    QQP|qqp)
        TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    QNLI|qnli)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        ;;
    MNLI|mnli)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
        ;;
    SNLI|snli)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        TASK_EXTRA="--max_seq_len 256 --num_sample 4"
        ;;
    trec)
        TEMPLATE="*cls**mask*:*+sent_0**sep+*"
        MAPPING="{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}"
        TASK_EXTRA="--first_sent_limit 110"
        ;;
    mr)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
        ;;
    cr)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
        ;;
    mpqa)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110"
        ;;
    CoLA|cola)
        TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
        MAPPING="{'0':'incorrect','1':'correct'}"
        ;;
    subj)
        TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
        MAPPING="{0:'subjective',1:'objective'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
        ;;
    MRPC|mrpc)
        TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    RTE|rte)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
        ;;
esac

ALL_ARGS_TOGETHER="
    --model_name_or_path $MODEL --few_shot_type $TYPE
    --task_name $TASK --template $TEMPLATE --mapping $MAPPING
    --data_dir $DATA_ROOT/$TASK/$K-$SEED
    --dataset_mode $DATASET_MODE
    --data_root $DATA_ROOT
    --full_dev_ratio $FULL_DEV_RATIO
    --overwrite_output_dir --output_dir $OUTPUT_DIR
    --num_k $K
    --tag $TAG
    --max_seq_length 128
    --seed $SEED
    --data_seed $DATA_SEED $SHUFFLE_FLAG
    --do_eval --do_predict --do_train
    --trainer $TRAINER
    --optimizer $OPT --max_steps $STEPS
    --logging_steps 10
    --per_device_eval_batch_size 4
    --evaluate_during_training
    --use_adaptive_h $USE_H
    --use_c_scale $USE_C
    $TASK_EXTRA
    $LOAD_KERNELS
    ${PY_ARGS[@]}
"

if [[ $NUM_GPU > 1 ]]; then
    # Randomly set a port number
    # If you encounter "address already used" error, just run again or manually set an available port id.
    PORT_ID=$(expr $RANDOM + 1000)

    # Allow multiple threads
    export OMP_NUM_THREADS=8
    python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID run.py \
        $ALL_ARGS_TOGETHER
else
    python run.py \
        $ALL_ARGS_TOGETHER
fi

run_exit_code=$?
rm -rf "$OUTPUT_DIR"/checkpoint-*
exit $run_exit_code
