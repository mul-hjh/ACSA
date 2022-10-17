import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn
import random
import numpy as np
import logging
import torchvision.transforms as transforms
from datasets.pedes import CuhkPedes
from models.model import Model
from utils import directory

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# 加载数据集
def data_config(image_dir, anno_dir, batch_size, split, max_length, transform):
    data_split = CuhkPedes(image_dir, anno_dir, split, max_length, transform)
    if split == 'train':
        shuffle = True
    else:
        shuffle = False
    loader = data.DataLoader(data_split, batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    return loader


# unique的含义是什么？
def get_image_unique(image_dir, anno_dir, batch_size, split, max_length, transform):
    if split == 'val':
        return CuhkPedes(image_dir, anno_dir, split, max_length, transform).unique_val
    else:  # test
        return CuhkPedes(image_dir, anno_dir, split, max_length, transform).unique_test


# 配置网络和优化器
def network_config(args, split='train', param=None, resume=False, model_path=None):
    network = Model(args)  # resnet模型
    network = nn.DataParallel(network).cuda()
    # print(len(network.state_dict().keys()))
    cudnn.benchmark = True
    # args.start_epoch = 0

    # process network params
    # 加载checkpoint, test或从断点处继续训练
    if resume:
        directory.check_file(model_path, 'model_file')
        checkpoint = torch.load(model_path)
        network_dict = checkpoint['network']
        # print(network_dict.keys())

        if split == 'train':
            args.start_epoch = checkpoint['epoch'] + 1
            network.load_state_dict(network_dict)
            print('==> Loading checkpoint "{}"'.format(model_path))

            ########## 查看每一层的名字 ##########
            dict_name = list(network.state_dict())
            for i, p in enumerate(dict_name):
                print(i, p)

            ########## 冻结指定层(bert)参数 ##########
            if args.frozen_bert:
                logging.info('==> Forzen bert！')
                for i, p in enumerate(network.parameters()):
                    if 193< i < 393:
                        p.requires_grad = False

            total = sum(p.numel() for p in network.parameters())
            logging.info('总参数个数：{}'.format(total))
            total_trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
            logging.info('需要训练的参数个数：{}'.format(total_trainable))

        elif split == 'val':
            network.load_state_dict(network_dict)
            print('==> Loading checkpoint "{}"'.format(model_path))

        else:  # 即test
            args.start_epoch = checkpoint['epoch'] + 1
            network.load_state_dict(network_dict)
            print('==> Loading checkpoint "{}"'.format(model_path))

    else:
        # 加载 pretrained_image_model
        if model_path is not None:
            network_dict = network.state_dict()  # resnet模型参数(初始化)
            # print("我们的模型所含模块： ", len(network_dict.keys()), network_dict.keys())  # 模型中所含的模块(如weight/bias等)及数量

            # 从预训练模型中加载Image相关的层
            # print(torch.load(model_path).keys())
            if args.image_model == 'resnet50':
                pretrained_image_model = torch.load(model_path)
            else:
                pretrained_image_model = torch.load(model_path)['model']  # ['model'] for swin transformer
            start = 0
            # print("pretrained module: ", pretrained_image_model.keys())
            # print(network_dict.keys())
            prefix = 'module.image_model.'  # 前缀, 对预训练模型的模块名称进行处理，使其与我们的模型对应
            pretrained_dict_image = {prefix + k[start:]: v for k, v in
                                     pretrained_image_model.items()}  # 处理来自imagenet的预训练模型
            pretrained_dict_image = {k: v for k, v in pretrained_dict_image.items() if
                                     k in network_dict}  # 将pretrained的模型与我们的模型进行比较，只保留相同的部分！
            logging.info('"==> Loading pretrained image model from: {}",  "加载的image_module个数: {}"'.format(model_path,
                                                                                                          len(
                                                                                                              pretrained_dict_image.keys())))

            ##################### 加载预训练模型中Text相关的层 #####################
            pretrained_language_model_path = '/home/junhua/pretrained_model/best_model/45.pth.tar'  # 从NAFS的best model里加载bert参数
            # pretrained_language_model_path = None
            if pretrained_language_model_path is not None:
                pretrained_language_model = torch.load(pretrained_language_model_path)
                pretrained_language_network = pretrained_language_model['network']
                # print(pretrained_network.keys())
                # 准备加载参数的层的名字
                text_layer_name = {'module.language_model.textExtractor.embeddings.word_embeddings.weight',
                                   'module.language_model.textExtractor.embeddings.position_embeddings.weight',
                                   'module.language_model.textExtractor.embeddings.token_type_embeddings.weight',
                                   'module.language_model.textExtractor.embeddings.LayerNorm.weight',
                                   'module.language_model.textExtractor.embeddings.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.0.attention.self.query.weight',
                                   'module.language_model.textExtractor.encoder.layer.0.attention.self.query.bias',
                                   'module.language_model.textExtractor.encoder.layer.0.attention.self.key.weight',
                                   'module.language_model.textExtractor.encoder.layer.0.attention.self.key.bias',
                                   'module.language_model.textExtractor.encoder.layer.0.attention.self.value.weight',
                                   'module.language_model.textExtractor.encoder.layer.0.attention.self.value.bias',
                                   'module.language_model.textExtractor.encoder.layer.0.attention.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.0.attention.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.0.attention.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.0.attention.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.0.intermediate.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.0.intermediate.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.0.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.0.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.0.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.0.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.1.attention.self.query.weight',
                                   'module.language_model.textExtractor.encoder.layer.1.attention.self.query.bias',
                                   'module.language_model.textExtractor.encoder.layer.1.attention.self.key.weight',
                                   'module.language_model.textExtractor.encoder.layer.1.attention.self.key.bias',
                                   'module.language_model.textExtractor.encoder.layer.1.attention.self.value.weight',
                                   'module.language_model.textExtractor.encoder.layer.1.attention.self.value.bias',
                                   'module.language_model.textExtractor.encoder.layer.1.attention.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.1.attention.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.1.attention.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.1.attention.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.1.intermediate.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.1.intermediate.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.1.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.1.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.1.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.1.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.2.attention.self.query.weight',
                                   'module.language_model.textExtractor.encoder.layer.2.attention.self.query.bias',
                                   'module.language_model.textExtractor.encoder.layer.2.attention.self.key.weight',
                                   'module.language_model.textExtractor.encoder.layer.2.attention.self.key.bias',
                                   'module.language_model.textExtractor.encoder.layer.2.attention.self.value.weight',
                                   'module.language_model.textExtractor.encoder.layer.2.attention.self.value.bias',
                                   'module.language_model.textExtractor.encoder.layer.2.attention.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.2.attention.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.2.attention.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.2.attention.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.2.intermediate.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.2.intermediate.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.2.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.2.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.2.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.2.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.3.attention.self.query.weight',
                                   'module.language_model.textExtractor.encoder.layer.3.attention.self.query.bias',
                                   'module.language_model.textExtractor.encoder.layer.3.attention.self.key.weight',
                                   'module.language_model.textExtractor.encoder.layer.3.attention.self.key.bias',
                                   'module.language_model.textExtractor.encoder.layer.3.attention.self.value.weight',
                                   'module.language_model.textExtractor.encoder.layer.3.attention.self.value.bias',
                                   'module.language_model.textExtractor.encoder.layer.3.attention.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.3.attention.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.3.attention.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.3.attention.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.3.intermediate.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.3.intermediate.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.3.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.3.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.3.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.3.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.4.attention.self.query.weight',
                                   'module.language_model.textExtractor.encoder.layer.4.attention.self.query.bias',
                                   'module.language_model.textExtractor.encoder.layer.4.attention.self.key.weight',
                                   'module.language_model.textExtractor.encoder.layer.4.attention.self.key.bias',
                                   'module.language_model.textExtractor.encoder.layer.4.attention.self.value.weight',
                                   'module.language_model.textExtractor.encoder.layer.4.attention.self.value.bias',
                                   'module.language_model.textExtractor.encoder.layer.4.attention.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.4.attention.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.4.attention.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.4.attention.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.4.intermediate.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.4.intermediate.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.4.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.4.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.4.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.4.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.5.attention.self.query.weight',
                                   'module.language_model.textExtractor.encoder.layer.5.attention.self.query.bias',
                                   'module.language_model.textExtractor.encoder.layer.5.attention.self.key.weight',
                                   'module.language_model.textExtractor.encoder.layer.5.attention.self.key.bias',
                                   'module.language_model.textExtractor.encoder.layer.5.attention.self.value.weight',
                                   'module.language_model.textExtractor.encoder.layer.5.attention.self.value.bias',
                                   'module.language_model.textExtractor.encoder.layer.5.attention.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.5.attention.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.5.attention.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.5.attention.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.5.intermediate.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.5.intermediate.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.5.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.5.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.5.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.5.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.6.attention.self.query.weight',
                                   'module.language_model.textExtractor.encoder.layer.6.attention.self.query.bias',
                                   'module.language_model.textExtractor.encoder.layer.6.attention.self.key.weight',
                                   'module.language_model.textExtractor.encoder.layer.6.attention.self.key.bias',
                                   'module.language_model.textExtractor.encoder.layer.6.attention.self.value.weight',
                                   'module.language_model.textExtractor.encoder.layer.6.attention.self.value.bias',
                                   'module.language_model.textExtractor.encoder.layer.6.attention.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.6.attention.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.6.attention.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.6.attention.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.6.intermediate.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.6.intermediate.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.6.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.6.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.6.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.6.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.7.attention.self.query.weight',
                                   'module.language_model.textExtractor.encoder.layer.7.attention.self.query.bias',
                                   'module.language_model.textExtractor.encoder.layer.7.attention.self.key.weight',
                                   'module.language_model.textExtractor.encoder.layer.7.attention.self.key.bias',
                                   'module.language_model.textExtractor.encoder.layer.7.attention.self.value.weight',
                                   'module.language_model.textExtractor.encoder.layer.7.attention.self.value.bias',
                                   'module.language_model.textExtractor.encoder.layer.7.attention.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.7.attention.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.7.attention.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.7.attention.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.7.intermediate.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.7.intermediate.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.7.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.7.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.7.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.7.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.8.attention.self.query.weight',
                                   'module.language_model.textExtractor.encoder.layer.8.attention.self.query.bias',
                                   'module.language_model.textExtractor.encoder.layer.8.attention.self.key.weight',
                                   'module.language_model.textExtractor.encoder.layer.8.attention.self.key.bias',
                                   'module.language_model.textExtractor.encoder.layer.8.attention.self.value.weight',
                                   'module.language_model.textExtractor.encoder.layer.8.attention.self.value.bias',
                                   'module.language_model.textExtractor.encoder.layer.8.attention.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.8.attention.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.8.attention.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.8.attention.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.8.intermediate.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.8.intermediate.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.8.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.8.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.8.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.8.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.9.attention.self.query.weight',
                                   'module.language_model.textExtractor.encoder.layer.9.attention.self.query.bias',
                                   'module.language_model.textExtractor.encoder.layer.9.attention.self.key.weight',
                                   'module.language_model.textExtractor.encoder.layer.9.attention.self.key.bias',
                                   'module.language_model.textExtractor.encoder.layer.9.attention.self.value.weight',
                                   'module.language_model.textExtractor.encoder.layer.9.attention.self.value.bias',
                                   'module.language_model.textExtractor.encoder.layer.9.attention.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.9.attention.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.9.attention.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.9.attention.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.9.intermediate.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.9.intermediate.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.9.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.9.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.9.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.9.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.10.attention.self.query.weight',
                                   'module.language_model.textExtractor.encoder.layer.10.attention.self.query.bias',
                                   'module.language_model.textExtractor.encoder.layer.10.attention.self.key.weight',
                                   'module.language_model.textExtractor.encoder.layer.10.attention.self.key.bias',
                                   'module.language_model.textExtractor.encoder.layer.10.attention.self.value.weight',
                                   'module.language_model.textExtractor.encoder.layer.10.attention.self.value.bias',
                                   'module.language_model.textExtractor.encoder.layer.10.attention.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.10.attention.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.10.attention.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.10.attention.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.10.intermediate.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.10.intermediate.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.10.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.10.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.10.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.10.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.11.attention.self.query.weight',
                                   'module.language_model.textExtractor.encoder.layer.11.attention.self.query.bias',
                                   'module.language_model.textExtractor.encoder.layer.11.attention.self.key.weight',
                                   'module.language_model.textExtractor.encoder.layer.11.attention.self.key.bias',
                                   'module.language_model.textExtractor.encoder.layer.11.attention.self.value.weight',
                                   'module.language_model.textExtractor.encoder.layer.11.attention.self.value.bias',
                                   'module.language_model.textExtractor.encoder.layer.11.attention.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.11.attention.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.11.attention.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.11.attention.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.encoder.layer.11.intermediate.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.11.intermediate.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.11.output.dense.weight',
                                   'module.language_model.textExtractor.encoder.layer.11.output.dense.bias',
                                   'module.language_model.textExtractor.encoder.layer.11.output.LayerNorm.weight',
                                   'module.language_model.textExtractor.encoder.layer.11.output.LayerNorm.bias',
                                   'module.language_model.textExtractor.pooler.dense.weight',
                                   'module.language_model.textExtractor.pooler.dense.bias'}
                text_layer_dict = {k: v for k, v in pretrained_language_network.items() if
                                   k in text_layer_name}  # 从模型中提取text相关的层
                text_layer_dict = {k: v for k, v in text_layer_dict.items() if
                                   k in network_dict}  # 将pretrained的模型与我们的模型进行比较，只保留相同的部分！
                logging.info('"==> Loading pretrained language model from: {}",  "加载的language_module个数: {}"'.format(
                    pretrained_language_model_path, len(text_layer_dict.keys())))
                network_dict.update(text_layer_dict)

            ######## 将预训练的参数更新到我们的模型中 ########
            network_dict.update(pretrained_dict_image)
            network.load_state_dict(network_dict)

            ########## 查看每一层的名字 ##########
            dict_name = list(network.state_dict())
            for i, p in enumerate(dict_name):
                print(i, p)

            ########## 冻结指定层(bert)参数 ##########
            if args.frozen_bert:
                logging.info('==> Forzen bert！')
                for i, p in enumerate(network.parameters()):
                    if 193 < i < 393:
                        p.requires_grad = False

            total = sum(p.numel() for p in network.parameters())
            logging.info('总参数个数：{}'.format(total))
            total_trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
            logging.info('需要训练的参数个数：{}'.format(total_trainable))
        else:
            logging.info("没有预训练模型")

    # process optimizer params 配置优化器
    if split == 'train':
        # optimizer
        # different params for different part
        image_params = list(map(id, network.module.image_model.parameters()))
        language_params = list(map(id, network.module.language_model.parameters()))
        transformer_params = image_params + language_params
        other_params = filter(lambda p: id(p) not in transformer_params, network.parameters())
        other_params = list(other_params)
        if param is not None:
            other_params.extend(list(param))
        # param_groups = [{'params': other_params, 'lr': 0.000001},
        param_groups = [{'params': other_params},   # , 'initial_lr':0.0003
                        {'params': network.module.image_model.parameters()}]    # , 'initial_lr':0.0003
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                lr=args.lr, betas=(args.adam_alpha, args.adam_beta), eps=args.epsilon, weight_decay=args.wd)
            print(optimizer)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(param_groups, eps=0.00000001, betas=(0.9, 0.999),
                                          lr=args.lr,
                                          weight_decay=0.05)  # Swin:  eps=1e-8, betas=(0.9, 0.999), lr=5e-4, weight_decay=0.05,min_lr=5e-6
        # print(optimizer)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9)
        if resume:
            print("==>Loading optimizer from checkpoint model!")
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # print(optimizer)

        # print(checkpoint.keys())
    else:
        optimizer = None

    logging.info('Total params: %2.fM' % (sum(p.numel() for p in network.parameters()) / 1000000.0))

    # seed
    #manualSeed = random.randint(1, 1000000000)
    manualSeed = int(0)
    logging.info('manualSeed:{}'.format(manualSeed))
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    return network, optimizer


def log_config(args, ca):
    filename = args.log_dir + '/' + ca + '.log'
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(handler)
    logging.info(args)


def dir_config(args):
    if not os.path.exists(args.image_dir):
        raise ValueError('Supply the dataset directory with --image_dir')
    if not os.path.exists(args.anno_dir):
        raise ValueError('Supply the anno file with --anno_dir')
    directory.makedir(args.log_dir)
    # save checkpoint
    directory.makedir(args.checkpoint_dir)
    directory.makedir(os.path.join(args.checkpoint_dir, 'model_best'))


# 设置lr在第几个epoch衰减（以下划线分隔，如50_100_150）
def lr_scheduler(optimizer, args):
    if '_' in args.epoches_decay:
        epoches_list = args.epoches_decay.split('_')
        epoches_list = [int(e) for e in epoches_list]
        gamma = 0.1  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if args.resume:
            last_epoch = args.last_epoch
        else:
            last_epoch = -1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, epoches_list, gamma,
                                                         last_epoch)  # 在特定的epoch进行衰减, 从断点继续训练时，last_epoch就等于加载模型的epoch！
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(args.epoches_decay))  # 等间隔（epoch）调整学习率
    return scheduler
