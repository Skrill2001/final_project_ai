# NOTE: use absolute path, modify according to your setting
CHECKPOINT_PATH="path to your checkpoint"
RESULT_PATH="path to the folder where you want to save results"

python -B inference.py --config-dir ./configs/ --config-name inference.yaml \
  dataset.gen_subset=test \
  common_eval.path=$CHECKPOINT_PATH \
  common_eval.results_path=$RESULT_PATH \
  override.modalities="['video','audio']" \
  common.user_dir=`pwd`