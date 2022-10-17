import os
import sys
import shutil
import datetime
import logging
import re
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.transforms as transforms
from utils.metric import AverageMeter, Loss
from test import test
from config import data_config, network_config, get_image_unique
from config import lr_scheduler
from train_config import config
from utils.scheduler_factory import create_scheduler
from models.bert import TextNet
from textblob import TextBlob
from tensorboardX import SummaryWriter

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def save_checkpoint(state, epoch, dst, is_best):
    filename = os.path.join(dst, str(args.start_epoch + epoch)) + '.pth.tar'  # 一定是args.start_epoch又变成0了，才会导致checkpoint覆盖，而非从21开始
    torch.save(state, filename)
    if is_best:
        dst_best = os.path.join(dst, 'model_best', str(args.start_epoch + epoch)) + '.pth.tar'
        shutil.copyfile(filename, dst_best)


# 这里的epoch已经+args.start_epoch了
def train(epoch, train_loader, network, optimizer, compute_loss, args):
    train_loss = AverageMeter()
    image_pre = AverageMeter()
    text_pre = AverageMeter()
    pos_sim = AverageMeter()
    neg_sim = AverageMeter()
    loss_path = os.path.join(args.checkpoint_dir, 'runs')
    loss_writer = SummaryWriter(loss_path)
    # switch to train mode
    network.train()
    
    avg_cmpm_loss = 0
    avg_cmpc_loss = 0
    avg_cont_loss = 0

    for step, (images, captions, labels) in enumerate(train_loader):
        images = images.cuda()  # torch.Size([batch_size, 3, 224, 224])
        labels = labels.cuda()

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
        # print(noun_phrases)

        tokens, segments, input_masks, caption_length = network.module.language_model.pre_process(captions, args.max_length)
        phrases_tokens, phrases_segments, phrases_input_masks, phrases_caption_length = network.module.language_model.pre_process(noun_phrases, 40)   # 将每个短语的最大长度限制到10个单词

        tokens = tokens.cuda()
        segments = segments.cuda()
        input_masks = input_masks.cuda()
        caption_length = caption_length.cuda()

        phrases_tokens = phrases_tokens.cuda()
        phrases_segments = phrases_segments.cuda()
        phrases_input_masks = phrases_input_masks.cuda()
        
        # compute loss
        image_features, multi_scale_text_features, image_query, image_value, text_key, text_value = network(images, tokens, segments, input_masks, phrases_tokens, phrases_segments, phrases_input_masks)  # bert

        cmpm_loss, cmpc_loss, cont_loss, loss, image_precision, text_precision, pos_avg_sim, neg_avg_sim, local_pos_avg_sim, local_neg_avg_sim = compute_loss(
            image_features, multi_scale_text_features, image_query, image_value, text_key, text_value, caption_length, labels)
        
        if step %100 == 0:
            loss_writer.add_scalar('Loss', loss, step)
            loss_writer.add_scalar('cmpm', cmpm_loss, step)
            loss_writer.add_scalar('cmpc', cmpc_loss, step)
            loss_writer.add_scalar('cont', cont_loss, step)

        # avg_cmpm_loss = avg_cmpm_loss + cmpm_loss.item()  # 加上这个可以避免显存逐渐增大
        # avg_cmpc_loss = avg_cmpc_loss + cmpc_loss.item()
        # avg_cont_loss = avg_cont_loss + cont_loss.item()
        
        # 每十个batch输出一次
        if step % 100 == 0:
            logging.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            logging.info('epoch:{}, step:{}, cmpm_loss:{:.3f}, cmpc_loss:{:.3f}, cont_loss:{:.3f}, local_pos_avg_sim:{:.3f}, local_neg_avg_sim:{:.3f}'
                         .format(epoch, step, cmpm_loss, cmpc_loss, cont_loss, local_pos_avg_sim * 100, local_neg_avg_sim * 100))
        
        # compute gradient and do ADAM step
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(network.parameters(), 5)
        optimizer.step()
        
        train_loss.update(loss, images.shape[0])
        image_pre.update(image_precision, images.shape[0])
        text_pre.update(text_precision, images.shape[0])
        pos_sim.update(pos_avg_sim, images.shape[0])
        neg_sim.update(neg_avg_sim, images.shape[0])
    
    return train_loss.avg, image_pre.avg, text_pre.avg, pos_sim.avg, neg_sim.avg


def main(args):
    # transform
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))    # ImageNet
        transforms.Normalize((0.391, 0.367, 0.350), (0.217, 0.212, 0.206))     # CUHK-PEDES
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))     # ImageNet
        transforms.Normalize((0.391, 0.367, 0.350), (0.217, 0.212, 0.206))     # CUHK-PEDES

    ])
    
    args.start_epoch = 0
    
    # data
    train_loader = data_config(args.image_dir, args.anno_dir, args.batch_size, 'train', args.max_length, train_transform)
    val_loader = data_config(args.image_dir, args.anno_dir, args.batch_size, 'val', args.max_length, val_transform)
    unique_image = get_image_unique(args.image_dir, args.anno_dir, args.batch_size, 'val', args.max_length, val_transform)
    
    # loss
    compute_loss = Loss(args)
    nn.DataParallel(compute_loss).cuda()
    
    # network
    network, optimizer = network_config(args, 'train', compute_loss.parameters(), args.resume, args.model_path)
    
    # lr_scheduler
    if args.CosineLRScheduler:
        scheduler = create_scheduler(args, optimizer)  # CosineLRScheduler
    else:
        scheduler = lr_scheduler(optimizer, args)  # MultiStepLR
    for param in optimizer.param_groups:
        logging.info('now_lr:{}'.format(param['lr']))
        break
    logging.info(scheduler.state_dict())
    
    # 迭代    resume之后，总的迭代次数不是num_epochs，而是要减去start_epoch;
    for epoch in range(args.num_epoches - args.start_epoch):
        # train for one epoch
        train_loss, image_precision, text_precision, pos_sim, neg_sim = train(args.start_epoch + epoch, train_loader, network, optimizer, compute_loss, args)
        # evaluate on validation set
        print('Train done for epoch-{}'.format(args.start_epoch + epoch))
        
        # save checkpoint
        state = {'network': network.state_dict(), 'optimizer': optimizer.state_dict(), 'W': compute_loss.W, 'epoch': args.start_epoch + epoch}
        save_checkpoint(state, epoch, args.checkpoint_dir, False)
        
        # save train logs
        logging.info('Epoch:  [{}|{}], train_loss: {:.3f}'.format(args.start_epoch + epoch, args.num_epoches, train_loss))
        logging.info(
            'image_precision: {:.3f}, text_precision: {:.3f}, pos_sim: {:.3f}, neg_sim: {:.3f}'.format(image_precision * 100, text_precision * 100, pos_sim * 100, neg_sim * 100))
        
        # validate this model

        
        # scheduler.step(args.start_epoch + epoch)  # CosineLR
        scheduler.step()      # StepLR
        for param in optimizer.param_groups:
            logging.info('now_lr:{}'.format(param['lr']))  # 此处输出的是下一个epoch要使用的lr！
            break
    
    logging.info('Train done')
    logging.info(args.checkpoint_dir)
    logging.info(args.log_dir)


if __name__ == "__main__":
    args = config()
    main(args)