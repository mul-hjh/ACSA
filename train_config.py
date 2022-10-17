import argparse
import os
import logging
from config import log_config, dir_config

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='command for train on CUHK-PEDES')
    """base root:
/home/junhua/DCMP-transformer/train.py
    """
    # Directory
    parser.add_argument('--image_dir', type=str, default='/home/junhua/Datasets/CUHK-PEDES/imgs', help='directory to store dataset')
    parser.add_argument('--anno_dir', type=str, default='/home/junhua/Datasets/CUHK-PEDES/processed_data/', help='directory to store anno file')
    
    parser.add_argument('--model_path', type=str, default='/home/junhua/pretrained_model/swin_tiny_patch4_window7_224.pth')
    #parser.add_argument('--model_path', type=str, default='/home/junhua/Head_upper_lower_feet/results/5+/model_best/29.pth.tar')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/junhua/Head_upper_lower_feet/results/7_3_2', help='directory to store checkpoint')
    parser.add_argument('--log_dir', type=str, default='/home/junhua/Head_upper_lower_feet/results/7_3_2/logs', help='directory to store log')
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser.add_argument('--num_patch', type=int, default=6)
    parser.add_argument('--num_phrase', type=int, default=10)
    parser.add_argument('--LocalToGlobal', default=True, action='store_true')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epoches', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--epoches_decay', type=str, default='8_15_20', help='#epoches when learning rate decays')

    parser.add_argument('--feature_size', type=int, default=768)
    parser.add_argument('--frozen_bert', default=True, action='store_true')
    parser.add_argument('--validate_frequency', type=int, default=500)      # 每隔几个epoch进行一次validate，以加快训练效率
    parser.add_argument('--reranking', default=True, action='store_true', help='whether reranking during testing')
    parser.add_argument('--max_length', type=int, default=118)


    # loss function settings
    parser.add_argument('--loss_weight', type=float, default=1)
    parser.add_argument('--CMPM', default=True, action='store_true')
    parser.add_argument('--CMPC', default=True, action='store_true')
    parser.add_argument('--CONT', default=True, action='store_true')    # metric.py
    parser.add_argument('--focal_type', type=str, default=None)     # metric.py
    parser.add_argument('--lambda_cont', type=float, default=0.1, help='hyper-parameter of contrastive loss')
    parser.add_argument('--lambda_softmax', type=float, default=20.0, help='scale constant')    # metric.py


    # image model
    parser.add_argument('--image_model', type=str, default='swintransformer', help='one of "resnet50", "swintransformer" ')
    parser.add_argument('--swin_type', type=str, default='tiny', help='one of "tiny", "small", "base" ')
    

    # optimizer
    parser.add_argument('--optimizer', type=str, default='adamw', help='one of "sgd", "adam", "adamw"')
    parser.add_argument('--CosineLRScheduler', default=False, action='store_true')
    parser.add_argument('--lr_min', type=float, default=0.05)     # lr_min*args.lr为最小的lr
    parser.add_argument('--warmup_init', type=float, default=0.05)     # warmup_init*args.lr为warm up的起始lr
    parser.add_argument('--warmup_epoch_num', type=int, default=5)     # warmup_init*args.lr为warm up的起始lr
    parser.add_argument('--wd', type=float, default=0.00004)

    # Resume
    parser.add_argument('--resume', default=False, action='store_true', help='whether or not to restore the pretrained whole model')
    parser.add_argument('--last_epoch', type=int, default=29)
    parser.add_argument('--pretrained', default=True, action='store_true', help='whether or not to restore the pretrained visual model')


    parser.add_argument('--ckpt_steps', type=int, default=5000, help='#steps to save checkpoint')
    parser.add_argument('--cnn_dropout_keep', type=float, default=0.999)
    parser.add_argument('--num_classes', type=int, default=11003)


    # Optimization setting
    parser.add_argument('--adam_alpha', type=float, default=0.9)
    parser.add_argument('--adam_beta', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--end_lr', type=float, default=1e-7, help='minimum end learning rate used by a polynomial decay learning rate')
    parser.add_argument('--lr_decay_type', type=str, default='exponential', help='One of "fixed" or "exponential"')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.9)


    # Default setting
    parser.add_argument('--gpus', type=str, default='1')

    args = parser.parse_args()
    return args


def config():
    args = parse_args()
    dir_config(args)
    log_config(args, 'train')
    return args
