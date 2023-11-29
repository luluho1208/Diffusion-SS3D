export CUDA_VISIBLE_DEVICES=$1
LOG_DIR=$2
DATA_RATIO=$3
LABELED_LIST=$4
CKPT=$5
python -u train.py --log_dir="${LOG_DIR}" --data_ratio="${DATA_RATIO}" --detector_checkpoint="${CKPT}" \
--labeled_sample_list="${LABELED_LIST}" --use_iou_for_nms --eval
