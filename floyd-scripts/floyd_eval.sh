export TRAIN_DIR="/output/"
export CHECK_PATH="/"
export PRETRAINED_MODEL_PATH="/floyd/input/pre-models/SqueezeDetPlus/SqueezeDetPlus.pkl"
export NET="squeezeDet+PruneFilter"

floyd run \
--data 2around/datasets/kitti/:dataset \
--data 2around/datasets/squeezedet-pre-models:pre-models \
--data 2around/datasets/filter_shape_checkpoints:checkpoints \
--env tensorflow-1.14 \
--gpu \
--max-runtime 600 \
  "python3 ./combined/src/GPU_eval.py \
  --dataset=KITTI \
  --image_set=val\
  --eval_dir=$TRAIN_DIR \
  --pretrained_model_path=$PRETRAINED_MODEL_PATH \
  --data_path=/floyd/input/dataset \
  --net=$NET \
  --checkpoint_path=$CHECK_PATH"
