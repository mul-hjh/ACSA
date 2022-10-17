import os
import sys
import shutil
import logging
import datetime
import re

import gc
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from utils.metric import AverageMeter, compute_topk
from test_config import config
from config import data_config, network_config, get_image_unique
from textblob import TextBlob
from tensorboardX import SummaryWriter

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def test(data_loader, network, args, unique_image):
    logging.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # switch to evaluate mode
    network.eval()
    max_size = args.batch_size * len(data_loader)
    images_bank = torch.zeros((max_size, args.feature_size)).cuda()
    text_bank = torch.zeros((max_size, args.feature_size)).cuda()
    labels_bank = torch.zeros(max_size).cuda()
    length_bank = torch.zeros(max_size, dtype=torch.long).cuda()
    index = 0

    img_query_bank = torch.zeros((max_size, 7, args.feature_size)).cuda()
    img_value_bank = torch.zeros((max_size, 7, args.feature_size)).cuda()

    text_key_bank = torch.zeros((max_size, args.num_phrase, args.feature_size)).cuda()
    text_value_bank = torch.zeros((max_size, args.num_phrase, args.feature_size)).cuda()

    with torch.no_grad():
        for images, captions, labels in data_loader:

            noun_phrases = []
            num_phrase = args.num_phrase
            for c in captions:
                c = TextBlob(c)
                phrases = c.noun_phrases
                if len(phrases) >= num_phrase:
                    noun_phrases = noun_phrases + phrases[0:num_phrase]
                else:
                    pad_length = num_phrase - len(phrases)
                    padding = ["[PAD]" for j in range(pad_length)]
                    noun_phrases = noun_phrases + phrases + padding

            tokens, segments, input_masks, caption_length = network.module.language_model.pre_process(captions, args.max_length)  # 这里的captions应该是tokenizer后的吧？bert是不是需要tokenizer前的？
            phrases_tokens, phrases_segments, phrases_input_masks, phrases_caption_length = network.module.language_model.pre_process(noun_phrases, 24)

            tokens = tokens.cuda()
            segments = segments.cuda()
            input_masks = input_masks.cuda()
            caption_length = caption_length.cuda()

            phrases_tokens = phrases_tokens.cuda()
            phrases_segments = phrases_segments.cuda()
            phrases_input_masks = phrases_input_masks.cuda()

            images = images.cuda()
            labels = labels.cuda()
            # patches = patches.cuda()
            interval = images.shape[0]

            # image_embeddings, text_embeddings = network(images, tokens, segments, input_masks)  # bert
            global_img_feat, global_text_feat, img_query, img_value, text_key, text_value = network(images, tokens,
                                                                                                    segments,
                                                                                                    input_masks,
                                                                                                    phrases_tokens,
                                                                                                    phrases_segments,
                                                                                                    phrases_input_masks)
            # global_img_feat, global_text_feat, img_query, img_value, text_key, text_value = network(images, tokens, segments, input_masks, sep_tokens, sep_segments, sep_input_masks, n_sep)

            images_bank[index: index + interval] = global_img_feat
            text_bank[index: index + interval] = global_text_feat
            labels_bank[index:index + interval] = labels

            img_query_bank[index: index + interval, :, :] = img_query
            img_value_bank[index: index + interval, :, :] = img_value
            text_key_bank[index: index + interval, :, :] = text_key
            text_value_bank[index: index + interval, :, :] = text_value
            length_bank[index:index + interval] = caption_length

            index = index + interval

        images_bank = images_bank[:index]
        text_bank = text_bank[:index]
        labels_bank = labels_bank[:index]
        length_bank = length_bank[:index]

        img_query_bank = img_query_bank[:index]
        img_value_bank = img_value_bank[:index]
        text_key_bank = text_key_bank[:index]
        text_value_bank = text_value_bank[:index]
        unique_image = torch.tensor(unique_image) == 1

        top1_1, top5_1, top10_1, top1_1_rerank, top5_1_rerank, top10_1_rerank, top1_2, top5_2, top10_2, top1_3, top5_3, top10_3, top1_4, top5_4, top10_4, top1_5, top5_5, top10_5, top1_6, top5_6, top10_6, top1_7, top5_7, top10_7 = compute_topk(
            images_bank[unique_image], img_query_bank[unique_image], img_value_bank[unique_image], text_bank,
            text_key_bank,
            text_value_bank, length_bank, labels_bank[unique_image], labels_bank, args, [1, 5, 10], True)

        return top1_1, top5_1, top10_1, top1_1_rerank, top5_1_rerank, top10_1_rerank, top1_2, top5_2, top10_2, top1_3, top5_3, top10_3, top1_4, top5_4, top10_4, top1_5, top5_5, top10_5, top1_6, top5_6, top10_6, top1_7, top5_7, top10_7


def main(args):
    # need to clear the pipeline
    # top1 & top10 need to be chosen in the same params ???
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transforms.Normalize((0.391, 0.367, 0.350), (0.217, 0.212, 0.206))     # CUHK-PEDES
    ])
    test_loader = data_config(args.image_dir, args.anno_dir, args.batch_size, 'test', args.max_length, test_transform)
    unique_image = get_image_unique(args.image_dir, args.anno_dir, args.batch_size, 'test', args.max_length,
                                    test_transform)
    acc_path = os.path.join(args.model_path, 'runs/acc')
    acc_writer = SummaryWriter(acc_path)

    top1_1 = 0.0
    top5_1 = 0.0
    top10_1 = 0.0
    top1_1_rerank = 0.0
    top5_1_rerank = 0.0
    top10_1_rerank = 0.0
    top1_2 = 0.0
    top5_2 = 0.0
    top10_2 = 0.0
    top1_3 = 0.0
    top5_3 = 0.0
    top10_3 = 0.0
    top1_4 = 0.0
    top5_4 = 0.0
    top10_4 = 0.0
    top1_5 = 0.0
    top5_5 = 0.0
    top10_5 = 0.0
    top1_6 = 0.0
    top5_6 = 0.0
    top10_6 = 0.0
    top1_7 = 0.0
    top5_7 = 0.0
    top10_7 = 0.0

    i2t_models = os.listdir(args.model_path)
    i2t_models.sort()
    i2t_models.reverse()
    #print(i2t_models)
    del i2t_models[0]
    del i2t_models[0]
    del i2t_models[0]
    # print(i2t_models[2].split('.')[0])
    i2t_models = sorted(i2t_models, key=lambda x: (int(x.split('.')[0])), reverse=False)
    # del i2t_models[0]      # 有个乱码文件，删掉
    print(i2t_models)
    iters = 0
    for i2t_model in i2t_models:
        model_file = os.path.join(args.model_path, i2t_model)
        if os.path.isdir(model_file):
            continue
        epoch = i2t_model.split('.')[0]
        network, _ = network_config(args, 'test', None, True, model_file)

        # print(network)
        top1_1, top5_1, top10_1, top1_1_rerank, top5_1_rerank, top10_1_rerank, top1_2, top5_2, top10_2, top1_3, top5_3, top10_3, top1_4, top5_4, top10_4, top1_5, top5_5, top10_5, top1_6, top5_6, top10_6, top1_7, top5_7, top10_7 = test(
            test_loader, network, args, unique_image)
        iters += 1
        acc_writer.add_scalars('acc', {'top1': top1_1, 'top5': top5_1, 'top10': top10_1}, global_step=iters)

        logging.info('epoch:{}'.format(epoch))
        logging.info(
            'top1_1: {:.3f}, top5_1: {:.3f}, top10_1: {:.3f}, top1_1_rerank: {:.3f}, top5_1_rerank: {:.3f}, top10_1_rerank: {:.3f}\n'
            'top1_2: {:.3f}, top5_2: {:.3f}, top10_2: {:.3f}, top1_3: {:.3f}, top5_3: {:.3f}, top10_3: {:.3f}\n'
            'top1_4: {:.3f}, top5_4: {:.3f}, top10_4: {:.3f}, top1_5: {:.3f}, top5_5: {:.3f}, top10_5: {:.3f}\n'
            'top1_6: {:.3f}, top5_6: {:.3f}, top10_6: {:.3f}, top1_7: {:.3f}, top5_7: {:.3f}, top10_7: {:.3f}'.format(
                top1_1, top5_1, top10_1, top1_1_rerank, top5_1_rerank, top10_1_rerank, top1_2, top5_2, top10_2, top1_3,
                top5_3, top10_3, top1_4, top5_4, top10_4, top1_5, top5_5, top10_5, top1_6, top5_6, top10_6, top1_7,
                top5_7, top10_7))

    logging.info(args.model_path)
    logging.info(args.log_dir)


if __name__ == '__main__':
    args = config()
    main(args)
