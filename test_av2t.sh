# NOTE: use absolute path, modify according to your setting
CHECKPOINT_PATH="/cpfs04/user/cuiziyun/rzd/exp/exp_a2t_nonlinear_encoder/checkpoints/checkpoint_best.pt"
RESULT_PATH="exp_a2t_nonlinear_encoder"

python -B inference.py --config-dir ./configs/ --config-name inference.yaml \
  dataset.gen_subset=test \
  common_eval.path=$CHECKPOINT_PATH \
  common_eval.results_path=$RESULT_PATH \
  override.modalities="['audio']" \
  common.user_dir=`pwd`

# override.modalities="['video','audio']" \