export CUDA_VISIBLE_DEVICES=$1
LOG_DIR=$2
DATA_RATIO=$3
LABELED_LIST=$4
PRETRAIN_CKPT=$5
python -u train.py --log_dir="${LOG_DIR}" --data_ratio="${DATA_RATIO}" --use_wandb \
--labeled_sample_list="${LABELED_LIST}" --detector_checkpoint="${PRETRAIN_CKPT}" --view_stats \
2>&1|tee "${LOG_DIR}"/LOG.log &