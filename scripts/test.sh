GPUS=1
export CUDA_VISIBLE_DEVICES=$GPUS

BASE_ROOT=/workspace2/junhua/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching-master
IMAGE_DIR=/workspace2/junhua/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching-master/datasets/imgs
ANNO_DIR=$BASE_ROOT/datasets/processed_data
CKPT_DIR=$BASE_ROOT/Results/model_data_December
LOG_DIR=$BASE_ROOT/Results/logs_December
#PRETRAINED_PATH=$BASE_ROOT/pretrained_models/mobilenet.tar
IMAGE_MODEL=resnet50
lr=0.0002
batch_size=256
lr_decay_ratio=0.9
# shellcheck disable=SC2034
epoches_decay=50_80_100


python3.5 ${BASE_ROOT}/test.py \
    --bidirectional \
    --model_path $CKPT_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --image_model $IMAGE_MODEL \
    --log_dir $LOG_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --image_dir $IMAGE_DIR \
    --anno_dir $ANNO_DIR \
    --gpus $GPUS \
    --epoch_ema 0
