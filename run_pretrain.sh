export CUDA_VISIBLE_DEVICES=$1
LOG_DIR=$2
DATA_RATIO=$3
LABELED_LIST=$4
python -u pretrain.py --log_dir="${LOG_DIR}" --data_ratio="${DATA_RATIO}" --use_wandb \
--labeled_sample_list="${LABELED_LIST}" 2>&1|tee "${LOG_DIR}"/LOG_ALL.log &