export TRAIN_DIR="/output"
export CHECK_PATH="/"
export PRETRAINED_MODEL_PATH="/floyd/input/pruned_weights/filter_shape/SqueezeDetPrunedFilterShape.pkl"
export NET="squeezeDet+PruneFilterShape"

floyd run \
--data 2around/datasets/kitti/:dataset \
--data 2around/datasets/pruned_weights:pruned_weights \
--env tensorflow-1.14 \
--gpu \
--max-runtime 600 \
 "python3 ./src/train.py \
  --pretrained_model_path=$PRETRAINED_MODEL_PATH \
  --data_path=/floyd/input/dataset \
  --image_set=train \
  --train_dir=$TRAIN_DIR \
  --max_steps=2000 \
  --net=$NET \
  --summary_step=100 \
  --checkpoint_step=200 \
  --checkpoint_dir=$CHECK_PATH \
  --pruning=False"

