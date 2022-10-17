import torch
import torch.nn as nn
from .resnet import resnet50
from .resnet import resnet101
from .resnet import resnet152
from models.swin_transformer import SwinTransformer
from models.bert import TextNet


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        out = out.squeeze(-1)
        out = out.squeeze(-1)

        return out


# ************选择图像/文本的特征提取网络***********
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        if args.image_model == 'resnet50':
            self.image_model = resnet50()
        elif args.image_model == 'swintransformer':
            self.image_model = SwinTransformer(args)
        
        self.visual_model = args.image_model  # swin-tiny和bert的输出都是768维，而resnet50是2048维，若使用则需要用1*1卷积块变换维度
        self.language_model = TextNet(args)

        #self.fc_window_stage2 = nn.Linear(192, 768)
        #self.fc_window_stage3 = nn.Linear(384, 768)

        self.num_patch = args.num_patch
        self.num_phrase = args.num_phrase

        self.basic_block = BasicBlock(768, 768, 1, None)
        self.window_conv = BasicBlock(384, 384, 1, None)

        self.fc_head = nn.Sequential()
        self.fc_head.add_module('fc_head', nn.Linear(256, 768))
        self.fc_head.add_module('relu_head', nn.ReLU(inplace=True))
        self.fc_head.add_module('drop_head', nn.Dropout(p=0.5))
        
        self.fc_upper = nn.Sequential()
        self.fc_upper.add_module('fc_upper', nn.Linear(256, 768))
        self.fc_upper.add_module('relu_upper', nn.ReLU(inplace=True))
        self.fc_upper.add_module('drop_upper', nn.Dropout(p=0.5))
        
        self.fc_lower = nn.Sequential()
        self.fc_lower.add_module('fc_lower', nn.Linear(256, 768))
        self.fc_lower.add_module('relu_lower', nn.ReLU(inplace=True))
        self.fc_lower.add_module('drop_lower', nn.Dropout(p=0.5))
        
        self.fc_feet = nn.Sequential()
        self.fc_feet.add_module('fc_feet', nn.Linear(128, 768))
        self.fc_feet.add_module('relu_feet', nn.ReLU(inplace=True))
        self.fc_feet.add_module('drop_feet', nn.Dropout(p=0.5))
        
        # self.fc1 = nn.Sequential()
        # self.fc1.add_module('fc1', nn.Linear(int(768/args.num_patch), 768))
        # self.fc1.add_module('relu1', nn.ReLU(inplace=True))
        # self.fc1.add_module('drop1', nn.Dropout(p=0.5))
        #
        # self.fc2 = nn.Sequential()
        # self.fc2.add_module('fc2', nn.Linear(768, 768))
        # self.fc2.add_module('relu2', nn.ReLU(inplace=True))
        # self.fc2.add_module('drop2', nn.Dropout(p=0.5))
        
        # BN layer before embedding projection
        self.bottleneck_image = nn.BatchNorm1d(args.feature_size)
        self.bottleneck_image.bias.requires_grad_(False)  # 冻结参数
        self.bottleneck_image.apply(weights_init_kaiming)  # 权重初始化
        
        self.bottleneck_text = nn.BatchNorm1d(args.feature_size)
        self.bottleneck_text.bias.requires_grad_(False)  # 冻结参数
        self.bottleneck_text.apply(weights_init_kaiming)  # 权重初始化
        
        # 构建text key/value
        self.fc_text_key = nn.Linear(768, args.feature_size)
        self.bottleneck_text_key = nn.LayerNorm([args.num_phrase, args.feature_size])
        self.fc_text_value = nn.Linear(768, args.feature_size)
        self.bottleneck_text_value = nn.LayerNorm([args.num_phrase, args.feature_size])
        
        # 构建image query/value
        self.fc_image_query = nn.Linear(args.feature_size, args.feature_size)
        self.fc_image_value = nn.Linear(args.feature_size, args.feature_size)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    
    # def forward(self, images, tokens, segments, input_masks, sep_tokens, sep_segments, sep_input_masks, n_sep):
    def forward(self, images, tokens, segments, input_masks, phrases_tokens, phrases_segments, phrases_input_masks):
        
        # 获得swin提取的image feature
        image_features, window_stage3, window_stage2 = self.image_model(images)  # transformer    stage3[4*bs, 384]
        # window_stage2 = self.fc_window_stage2(window_stage2).view(-1, 16, 768)
        #print(window_stage2.size())

        image_features = self.bottleneck_image(image_features)    # [bs, 768]
        window_stage3 = self.window_conv(window_stage3)    # [4*bs, 384]
        window_stage3 = window_stage3.view(-1, 2, 768)

        local_features = self.basic_block(image_features)
        local_features = local_features.view(-1, self.num_patch, int(768/self.num_patch))    #
        
        head = local_features[:, :2, :].view(-1, 1, 256)    # torch.Size([bs, 2, 128])
        head = self.fc_head(head)    # torch.Size([bs, 1, 768])
        upper_body = local_features[:, 1:3, :].view(-1, 1, 256)    # torch.Size([bs, 2, 128])
        upper_body = self.fc_upper(upper_body)    # torch.Size([bs, 1, 768])
        lower_body = local_features[:, 3:5, :].view(-1, 1, 256)    #torch.Size([bs, 2, 128])
        lower_body = self.fc_lower(lower_body)    # torch.Size([bs, 1, 768])
        feet = local_features[:, 5:, :]        #torch.Size([bs, 1, 128])
        feet = self.fc_feet(feet)    # torch.Size([bs, 1, 768])

        local_embeddings = torch.cat((head, upper_body, lower_body, feet, window_stage3), dim=1)    # torch.Size([bs, 4, 768])

        global_embeddings = local_embeddings.permute(0, 2, 1)
        global_embeddings = self.avgpool(global_embeddings).squeeze(-1)
        multi_scale_window_feature = torch.cat((global_embeddings.unsqueeze(1), local_embeddings), dim=1)  # torch.Size([bs, 21, 768]) 这是将不同stage的window feature和global feature合并到一起了

        image_query = self.fc_image_query(multi_scale_window_feature)
        image_value = self.fc_image_value(multi_scale_window_feature)
        
        # 获得multi scale text feature
        text_features = self.language_model(tokens, segments, input_masks)  # sentences
        global_text_feat = text_features[:, 0]
        global_text_feat = self.bottleneck_text(global_text_feat)

        phrases_features = self.language_model(phrases_tokens, phrases_segments, phrases_input_masks)  # noun phrases
        phrases_features = phrases_features[:, 0, :]
        phrases_features = phrases_features.view(-1, self.num_phrase, phrases_features.size(1))    # 一个句子中最多含的名词短语数量

        # word_features = text_features[:, 1:99]  # words

        #multi_scale_text_features = torch.cat((phrases_features, word_features), dim=1)  # concat phrases and words
        #multi_scale_text_features = torch.cat((global_text_feat.unsqueeze(1), phrases_features), dim=1)  # concat phrases and global text feature
        # multi_scale_text_features = word_features    # only words
        multi_scale_text_features = phrases_features    # only noun phrases

        text_key = self.fc_text_key(multi_scale_text_features)
        text_key = self.bottleneck_text_key(text_key)
        text_value = self.fc_text_value(multi_scale_text_features)
        text_value = self.bottleneck_text_value(text_value)
        
        # print(image_features.size(), global_text_feat.size())
        # print(image_query.size(), text_value.size())
        
        return image_features, global_text_feat, image_query, image_value, text_key, text_value     # torch.Size([batch_size, args.feature_size])
