# NOTE: use absolute path, modify according to your setting
DATA_DIR="/cpfs04/user/cuiziyun/rzd/final_project_data/30h_data"
TOKENIZER_PATH="/cpfs04/user/cuiziyun/rzd/final_project_data/spm1000/spm_unigram1000.model"
PRETRAIN_PATH="/cpfs04/user/cuiziyun/rzd/final_project_ckpt/pretrained_model.pth"

export PYTHONPATH=./:$PYTHONPATH
python -u main.py --config-dir configs/ \
  --config-name audiovideo2text.yaml \
  task.data=$DATA_DIR \
  task.label_dir=$DATA_DIR \
  task.tokenizer_bpe_model=$TOKENIZER_PATH \
  task.modalities="['audio']" \
  model.pretrained_path=$PRETRAIN_PATH \
  hydra.run.dir="./exp/exp_a2t_nonlinear_encoder" common.user_dir=`pwd`

  # task.modalities="['audio', 'video']" \