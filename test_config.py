import argparse
from config import log_config
import logging
import os


def parse_args():
    parser = argparse.ArgumentParser(description='command for evaluate on CUHK-PEDES')
    # Directory
    parser.add_argument('--image_dir',
                        default='/home/junhua/Datasets/CUHK-PEDES/imgs',
                        type=str, help='directory to store dataset')
    parser.add_argument('--anno_dir',
                        default='/home/junhua/Datasets/CUHK-PEDES/processed_data/',
                        type=str, help='directory to store anno')
    parser.add_argument('--model_path', type=str,
                        default='/home/junhua/Head_upper_lower_feet/results/7_3_2',
                        help='directory to load checkpoint')
    parser.add_argument('--log_dir', type=str,
                        default='/home/junhua/Head_upper_lower_feet/results/7_3_2/logs',
                        help='directory to store log')
   
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser.add_argument('--num_patch', type=int, default=6)
    parser.add_argument('--num_phrase', type=int, default=10)
    parser.add_argument('--LocalToGlobal', default=True, action='store_true')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_model', type=str, default='swintransformer')
    parser.add_argument('--swin_type', type=str, default='tiny')
    parser.add_argument('--feature_size', type=int, default=768)
    parser.add_argument('--reranking', default=True, action='store_true', help='whether reranking during testing')
    parser.add_argument('--focal_type', type=str, default=None)
    parser.add_argument('--lambda_softmax', type=float, default=20.0, help='scale constant')

    # SSL
    parser.add_argument('--isjigsaw', default=False, action='store_true', help='use Jigsaw Puzzle as pretext-task')
    parser.add_argument('--jigsaw_classifier', default=False, action='store_true', help='if multi-task learning, you need this')    # 为什么这个参数没有传过去？————注意位置参数和关键字参数的区别

    # LSTM setting
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--num_lstm_units', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=12000)
    parser.add_argument('--lstm_dropout_ratio', type=float, default=0.7)
    parser.add_argument('--bidirectional', default=True, action='store_true')

    parser.add_argument('--max_length', type=int, default=118)

    # parser.add_argument('--image_model', type=str, default='mobilenet_v1')
    parser.add_argument('--cnn_dropout_keep', type=float, default=0.999)

    # Default setting
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    return args


def config():
    args = parse_args()
    log_config(args, 'test')
    return args
