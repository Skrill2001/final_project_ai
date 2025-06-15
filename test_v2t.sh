# NOTE: use absolute path, modify according to your setting
CHECKPOINT_PATH="/cpfs02/user/housiyuan/project/final_project_ai/exp/exp_v2t/checkpoints/checkpoint_best.pt"
RESULT_PATH="test_results"

python -B inference.py --config-dir ./configs/ --config-name inference.yaml \
  dataset.gen_subset=test \
  common_eval.path=$CHECKPOINT_PATH \
  common_eval.results_path=$RESULT_PATH \
  override.modalities="['video']" \
  common.user_dir=`pwd`