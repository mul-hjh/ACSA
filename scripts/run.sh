GPUS=0
export CUDA_VISIBLE_DEVICES=$GPUS

BASE_ROOT=/workspace2/junhua/DCMP-transformer
IMAGE_DIR=/workspace2/junhua/DCMP-Rotation/datasets/imgs
ANNO_DIR=/workspace2/junhua/DCMP-transformer/datasets/processed_data

CKPT_DIR=/workspace2/junhua/DCMP-transformer/result/demo/
LOG_DIR=/workspace2/junhua/DCMP-transformer/result/demo/logs

PRETRAINED_PATH=/workspace2/junhua/DCMP-transformer/pretrained_model/resnet50-19c8e357.pth
#PRETRAINED_PATH=
IMAGE_MODEL=transformer
lr=0.0002
lr_decay_ratio=0.9

batch_size=32
ssl_weight=0

num_epoches=40
epoches_decay=5_25_40
last_epoch=63

python3.5 $BASE_ROOT/train.py \
    --last_epoch $last_epoch \
    --CMPM \
    --CMPC \
    --bidirectional \
    --pretrained \
    --ssl_weight $ssl_weight \
    --model_path $PRETRAINED_PATH \
    --image_model $IMAGE_MODEL \
    --log_dir $LOG_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --checkpoint_dir $CKPT_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --image_dir $IMAGE_DIR \
    --anno_dir $ANNO_DIR \
    --batch_size $batch_size \
    --gpus $GPUS \
    --num_epoches $num_epoches \
    --lr $lr \
    --lr_decay_ratio $lr_decay_ratio \
    --epoches_decay ${epoches_decay}


python3.5 $BASE_ROOT/test.py \
    --bidirectional \
    --model_path $CKPT_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --image_model $IMAGE_MODEL \
    --log_dir $LOG_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --image_dir $IMAGE_DIR \
    --anno_dir $ANNO_DIR \
    --gpus $GPUS \
    --epoch_ema 0
