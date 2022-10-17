BASE_ROOT=/workspace2/junhua/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching-master

IMAGE_ROOT=$BASE_ROOT/datasets/imgs
JSON_ROOT=$BASE_ROOT/datasets/reid_raw.json
OUT_ROOT=$BASE_ROOT/datasets/processed_data


echo "Process CUHK-PEDES dataset and save it as pickle form"

python ${BASE_ROOT}/datasets/preprocess.py \
        --img_root=${IMAGE_ROOT} \
        --json_root=${JSON_ROOT} \
        --out_root=${OUT_ROOT} \
        --min_word_count 3 
