import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from train_config import config

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class FocalLoss(nn.Module):
	
	def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.eps = eps
		self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)
	
	def forward(self, input, target):
		logp = self.ce(input, target)
		pt = torch.exp(-logp)
		loss = (1 - pt) ** self.gamma * logp  # alpha=0.25
		
		return loss.mean()


def l2norm(X, dim, eps=1e-8):
	"""L2-normalize columns of X
    """
	norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
	X = torch.div(X, norm)
	return X


class EMA():
	def __init__(self, decay=0.999):
		self.decay = decay
		self.shadow = {}
	
	def register(self, name, val):
		self.shadow[name] = val.cpu().detach()
	
	def get(self, name):
		return self.shadow[name]
	
	def update(self, name, x):
		assert name in self.shadow
		new_average = (1.0 - self.decay) * x.cpu().detach() + self.decay * self.shadow[name]
		self.shadow[name] = new_average.clone()


def pairwise_distance(A, B):
	"""
    Compute distance between points in A and points in B
    :param A:  (m,n) -m points, each of n dimension. Every row vector is a point, denoted as A(i).
    :param B:  (k,n) -k points, each of n dimension. Every row vector is a point, denoted as B(j).
    :return:  Matrix with (m, k). And the ele in (i,j) is the distance between A(i) and B(j)
    """
	A_square = torch.sum(A * A, dim=1, keepdim=True)
	B_square = torch.sum(B * B, dim=1, keepdim=True)
	
	distance = A_square + B_square.t() - 2 * torch.matmul(A, B.t())
	
	return distance


def one_hot_coding(index, k):
	if type(index) is torch.Tensor:
		length = len(index)
	else:
		length = 1
	out = torch.zeros((length, k), dtype=torch.int64).cuda()
	index = index.reshape((len(index), 1))
	out.scatter_(1, index, 1)
	return out


def compute_similarity(x1, x2, dim=1, eps=1e-8):
	"""Returns cosine similarity between x1 and x2, computed along dim."""
	w12 = torch.sum(x1 * x2, dim)
	w1 = torch.norm(x1, 2, dim)
	w2 = torch.norm(x2, 2, dim)
	return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


# for ’compute_weiTexts‘, img和word互为value，双向计算attention的值，下面的备注均以i2t为例
# local_img_query[bs, 6, 768], local_img_value[bs, 6, 768], local_text_key[bs, 101, 768], local_text_value[bs, 101, 768]
def func_attention_MxN(local_img_query, txt_i_key_expand, txt_i_value_expand, opt, eps=1e-8):  # 输入的名字仅供参考，位置才是重要的
	"""
    query: (batch, queryL, d)       (bs, d)
    context: (batch, sourceL, d)    (bs, d)
    opt: parameters
    """
	# txt_i_key_expand: (n_img, n_word, d)
	batch_size, queryL, sourceL = txt_i_key_expand.size(0), local_img_query.size(1), txt_i_key_expand.size(1)  # bs, 6, n_word
	local_img_query_norm = l2norm(local_img_query, dim=-1)
	txt_i_key_expand_norm = l2norm(txt_i_key_expand, dim=-1)
	
	# Step 1: pre assign attention    ————计算权重
	# --> (batch, d, queryL)
	local_img_queryT = torch.transpose(local_img_query_norm, 1, 2)  # torch.Size([bs, n_word, 6])
	
	# (bs, sourceL, d) * (bs, d, queryL) --> (bs, sourceL, queryL)
	attn = torch.bmm(txt_i_key_expand_norm, local_img_queryT)  # bmm是一个相乘操作，且二者必须都是3个维度（换成global query的话是不是把这里换成matmul就可以了？
	attn = nn.LeakyReLU(0.1)(attn)
	attn = l2norm(attn, 2)
	
	# --> (batch, queryL, sourceL)
	attn = torch.transpose(attn, 1, 2).contiguous()
	# --> (batch*queryL, sourceL)
	attn = attn.view(batch_size * queryL, sourceL)
	
	attn = nn.Softmax(dim=1)(attn * opt.lambda_softmax)
	# --> (batch, queryL, sourceL)
	attn = attn.view(batch_size, queryL, sourceL)
	# print('attn: ', attn)
	
	# Step 2: identify irrelevant fragments   ————————————这一步是必要的吗？ 似乎不是！可以试一下每种选择效果都如何
	# Learning an indicator function H, one for relevant, zero for irrelevant   学习指标函数H，1表示相关，0表示不相关
	if opt.focal_type == 'equal':
		funcH = focal_equal(attn, batch_size, queryL, sourceL)
	elif opt.focal_type == 'prob':
		funcH = focal_prob(attn, batch_size, queryL, sourceL)
	else:
		funcH = None
	
	# Step 3: reassign attention    ————给value加权
	if funcH is not None:
		tmp_attn = funcH * attn
		attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)
		attn = tmp_attn / attn_sum
	
	# --> (batch, d, sourceL)
	txt_i_valueT = torch.transpose(txt_i_value_expand, 1, 2)
	# --> (batch, sourceL, queryL)
	attnT = torch.transpose(attn, 1, 2).contiguous()
	
	# (bs, d, sourceL) * (bs, sourceL, queryL) --> (bs, d, queryL)
	weightedContext = torch.bmm(txt_i_valueT, attnT)
	# 换位 --> (batch, queryL, d)
	weightedContext = torch.transpose(weightedContext, 1, 2)
	
	# return weightedContext, attn
	return weightedContext


def focal_equal(attn, batch_size, queryL, sourceL):
	"""
    consider the confidence g(x) for each fragment as equal
    sigma_{j} (xi - xj) = sigma_{j} xi - sigma_{j} xj
    attn: (batch, queryL, sourceL)
    """
	funcF = attn * sourceL - torch.sum(attn, dim=-1, keepdim=True)
	fattn = torch.where(funcF > 0, torch.ones_like(attn),
	                    torch.zeros_like(attn))
	return fattn


def focal_prob(attn, batch_size, queryL, sourceL):
	"""
    consider the confidence g(x) for each fragment as the sqrt of their similarity probability to the query fragment
    sigma_{j} (xi - xj)gj = sigma_{j} xi*gj - sigma_{j} xj*gj
    attn: (batch, queryL, sourceL)
    """
	
	# -> (batch, queryL, sourceL, 1)
	xi = attn.unsqueeze(-1).contiguous()
	# -> (batch, queryL, 1, sourceL)
	xj = attn.unsqueeze(2).contiguous()
	# -> (batch, queryL, 1, sourceL)
	xj_confi = torch.sqrt(xj)
	
	xi = xi.view(batch_size * queryL, sourceL, 1)
	xj = xj.view(batch_size * queryL, 1, sourceL)
	xj_confi = xj_confi.view(batch_size * queryL, 1, sourceL)
	
	# -> (batch*queryL, sourceL, sourceL)
	term1 = torch.bmm(xi, xj_confi)
	term2 = xj * xj_confi
	funcF = torch.sum(term1 - term2, dim=-1)  # -> (batch*queryL, sourceL)
	funcF = funcF.view(batch_size, queryL, sourceL)
	
	fattn = torch.where(funcF > 0, torch.ones_like(attn),
	                    torch.zeros_like(attn))
	return fattn


"""对于只有成对对应的匹配任务，我们可以利用cmpm损失来学习有区别的图像-文本嵌入; 如果身份标签可用，我们联合cmpm和cmpc损失来更准确地关联跨模态表示。"""


class Loss(nn.Module):
	def __init__(self, args):
		super(Loss, self).__init__()
		self.args = args
		self.CMPM = args.CMPM
		self.CMPC = args.CMPC
		self.CONT = args.CONT
		self.epsilon = args.epsilon
		self.num_classes = args.num_classes
		self.avgpool = nn.AdaptiveAvgPool1d(1)
		
		if args.resume:
			checkpoint = torch.load(args.model_path)
			self.W = Parameter(checkpoint['W'])  # compute_loss.W
			print('=========> Loading in parameter W from pretrained models')
		else:
			self.W = Parameter(torch.randn(args.feature_size, args.num_classes))  # ??????
			self.init_weight()
	
	def init_weight(self):
		nn.init.xavier_uniform_(self.W.data, gain=1)  # xavier是一种初始化方法
	
	@staticmethod  # 给每个word赋予权重，计算加权后的相似度
	def compute_weiTexts(image_query, image_value, text_key, text_value, text_length, args):
		"""
        Compute weighted text embeddings
        :param image_embeddings: Tensor with dtype torch.float32, [n_img, n_region, d]
        :param text_embeddings: Tensor with dtype torch.float32, [n_txt, n_word, d]
        :param text_length: list, contain length of each sentence, [batch_size]
        :param labels: Tensor with dtype torch.int32, [batch_size]
        :return: i2t_similarities: Tensor, [n_img, n_txt]
                 t2i_similarities: Tensor, [n_img, n_txt]
        """
		# local_img_query[bs, 6, 768], local_img_value[bs, 6, 768], local_text_key[bs, 101, 768], local_text_value[bs, 101, 768]
		# image_query = image_query.unsqueeze(1)
		# image_value = image_value.unsqueeze(1)
		# print(image_query.size(), image_value.size(), text_key.size(), text_value.size(), "++++++++++")    # torch.Size([bs, 768])
		n_img = image_query.shape[0]  # batch_size
		n_txt = text_key.shape[0]  # batch_size
		t2i_similarities = []
		i2t_similarities = []
		# atten_final_result = {}
		for i in range(n_txt):
			# Get the i-th text description
			n_word = text_length[i]  # bs个句子，每个句子有 n_word 个word
			txt_i_key = text_key[i, :n_word, :].unsqueeze(0).contiguous()  # torch.Size([1, n_word, d])   d=feature_size 每个caption的word-level feature，每个word的特征都是768维
			txt_i_value = text_value[i, :n_word, :].unsqueeze(0).contiguous()  # # torch.Size([1, n_word, d])
			# -> (n_img, n_word, d)
			txt_i_key_expand = txt_i_key.repeat(n_img, 1, 1)  # 复制bs行, 即每个caption与所有的image embedding计算attention
			txt_i_value_expand = txt_i_value.repeat(n_img, 1, 1)  # 复制bs行, 即每个caption与所有的image embedding计算attention
			
			# -> (n_img, n_region, d)
			# 对每个word赋予权重, 得到image query对应的加权文本表示
			weiText = func_attention_MxN(image_query, txt_i_key_expand, txt_i_value_expand, args)  # torch.Size([bs, 6, 768])    [bs, d]
			# print(weiText.size())
			# atten_final_result[i] = atten_text[i, :, :]
			# image_embeddings = l2norm(image_embeddings, dim=2)
			weiText = l2norm(weiText, dim=2)
			i2t_sim = compute_similarity(image_value, weiText, dim=2)  # [bs, queryL, d] * [bs, queryL, d] --> torch.Size([bs, 6])    # [bs, d] * [bs, d] --> [bs]
			# print(i2t_sim.size())
			# if len(i2t_sim.size()) == 1:  # test/val:2    train:1     # 这两句是把swin 提取的特征直接作为local feature时所加的
			#     i2t_sim = i2t_sim.unsqueeze(1)  # torch.Size([200, 1, 6])
			i2t_sim = i2t_sim.mean(dim=1, keepdim=True)  # torch.Size([bs, 1])
			# i2t_sim = i2t_sim.unsqueeze(-1)    # [bs, 1]
			i2t_similarities.append(i2t_sim)
			
			# -> (n_img, n_word, d)
			# weiImage, atten_image = func_attention_MxN(txt_i_key_expand, local_img_query, local_img_value, args)
			# 对每个visual component求权重，得到每个word对应的加权视觉表示
			weiImage = func_attention_MxN(txt_i_key_expand, image_query, image_value, args)  # torch.Size([bs, d])
			# print(weiImage.size())
			# txt_i_expand = l2norm(txt_i_expand, dim=2)
			weiImage = l2norm(weiImage, dim=2)
			t2i_sim = compute_similarity(txt_i_value_expand, weiImage, dim=2)  # torch.Size([bs])
			t2i_sim = t2i_sim.mean(dim=1, keepdim=True)
			# t2i_sim = t2i_sim.unsqueeze(-1)
			t2i_similarities.append(t2i_sim)  # list
			# print(type(t2i_sim), type(t2i_similarities))
		
		# (n_img, n_txt)
		# torch.save(atten_final_result, 'atten_final_result.pt')
		# i2t_similarities = torch.tensor(i2t_similarities)
		# print(type(i2t_similarities))
		# print(i2t_similarities.shape)
		i2t_similarities = torch.cat(i2t_similarities, 1)  # torch.Size([bs, bs])
		t2i_similarities = torch.cat(t2i_similarities, 1)  # torch.Size([bs, bs])
		# print(i2t_similarities.size(), t2i_similarities.size())
		# print(weiText.size())
		return i2t_similarities, t2i_similarities
		# return weiImage, weiText
	
	def contrastive_loss(self, i2t_similarities, t2i_similarities, labels):
		batch_size = i2t_similarities.shape[0]
		labels_reshape = torch.reshape(labels, (batch_size, 1))
		labels_dist = labels_reshape - labels_reshape.t()
		labels_mask = (labels_dist == 0)
		# criterion = FocalLoss(gamma=2)
		
		# normalize the true matching distribution
		labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)
		
		i2t_pred = F.softmax(i2t_similarities * self.args.lambda_softmax, dim=1)
		i2t_loss = i2t_pred * (F.log_softmax(i2t_similarities * self.args.lambda_softmax, dim=1) - torch.log(labels_mask_norm + self.epsilon))
		
		t2i_pred = F.softmax(t2i_similarities * self.args.lambda_softmax, dim=1)
		t2i_loss = t2i_pred * (F.log_softmax(t2i_similarities * self.args.lambda_softmax, dim=1) - torch.log(labels_mask_norm + self.epsilon))
		
		constrastive_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
		
		sim_cos = i2t_similarities
		pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
		neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))
		
		return constrastive_loss, pos_avg_sim, neg_avg_sim
		# return constrastive_loss
	
	# CMPC LOSS定义
	def compute_cmpc_loss(self, image_embeddings, text_embeddings, labels):
		
		"""
        Cross-Modal Projection Classfication loss(CMPC)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:  image_precision: precision of image classification;    text_precision: precision of text classification
        """
		
		criterion = nn.CrossEntropyLoss(reduction='mean')       # 自动把target变成one-hot形式
		# criterion = FocalLoss(gamma=2)  # 自动把target变成one-hot形式
		
		self.W_norm = self.W / self.W.norm(dim=0)  # 权重归一化
		
		image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)  # 公式（8）
		text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
		
		image_proj_text = torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm  # 跨模态投影
		text_proj_image = torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm
		
		image_logits = torch.matmul(image_proj_text, self.W_norm)
		# image_logits = F.log_softmax(image_logits, dim=1)             # softmax
		# print(image_logits.sum(1))
		text_logits = torch.matmul(text_proj_image, self.W_norm)
		# text_logits = F.log_softmax(text_logits, dim=1)               # softmax
		# print(text_logits.sum(1))
		
		# labels_onehot = one_hot_coding(labels, self.num_classes).float()
		# cmpc_loss = -(image_logits + text_logits) * labels_onehot
		# cmpc_loss = torch.mean(torch.sum(cmpc_loss, dim=1))
		cmpc_loss = criterion(image_logits, labels) + criterion(text_logits, labels)  #
		
		# classification accuracy for observation
		image_pred = torch.argmax(image_logits, dim=1)
		text_pred = torch.argmax(text_logits, dim=1)
		
		image_precision = torch.mean((image_pred == labels).float())
		text_precision = torch.mean((text_pred == labels).float())
		
		return cmpc_loss, image_precision, text_precision
	
	# CMPM LOSS定义, cmpm旨在增加不匹配样本之间的方差以及匹配样本之间的关联！————即减小负对的兼容性，增大正对的相关性
	def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
		"""
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: average cosine-similarity for negative pairs
        """
		
		"""
        数据是以pair的形式（with label）输入的！！！ debug此处可以看清输入数据
        """
		batch_size = image_embeddings.shape[0]
		labels_reshape = torch.reshape(labels, (batch_size, 1))  # torch.Size([batch_size, 1])
		labels_dist = labels_reshape - labels_reshape.t()  # batch_size * batch_size的矩阵，对角线是0
		labels_mask = (labels_dist == 0)  # 同上，只有对角线是1,其余位置全是0
		
		image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)  # i / ||i||
		text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
		
		image_proj_text = torch.matmul(image_embeddings, text_norm.t())  # image向量 在text上的投影，投影越长，越相似（如paper Fig.2 (a)）
		text_proj_image = torch.matmul(text_embeddings, image_norm.t())  # text向量 在image上的投影
		
		# normalize the true matching distribution
		labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)  # paper：公式（2）  qij
		
		i2t_pred = F.softmax(image_proj_text, dim=1)  # 公式（1）
		i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon))  # 公式（3）
		
		t2i_pred = F.softmax(text_proj_image, dim=1)
		t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))
		
		cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))  # 公式（4）+ （5）
		
		sim_cos = torch.matmul(image_norm, text_norm.t())  # 余弦相似度
		
		pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))  # masked_select返回一个mask后的一维向量，只保留对角线的batch_size个值，即正对的相似度
		neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))  # mask后保留除对角线之外的batch_size*batch_size - batch_size个值，即负对的相似度
		
		return cmpm_loss, pos_avg_sim, neg_avg_sim
	
	# 计算loss和准确率
	def forward(self, image_embeddings, text_embeddings, image_query, image_value, text_key, text_value, text_length, labels):
		cmpm_loss = 0.0
		cmpc_loss = 0.0
		cont_loss = 0.0
		image_precision = 0.0
		text_precision = 0.0
		neg_avg_sim = 0.0
		pos_avg_sim = 0.0
		local_pos_avg_sim = 0.0
		local_neg_avg_sim = 0.0
		if self.CMPM:
			# text_embeddings = self.avgpool(text_embeddings.transpose(1, 2))  # B C 1
			# text_embeddings = torch.flatten(text_embeddings, 1)  # torch.Size([8, 768])
			# image_feature, text_feature = self.compute_weiTexts(image_query, image_value, text_key, text_value, text_length, self.args)
			cmpm_loss, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(image_embeddings, text_embeddings, labels)
		if self.CMPC:
			# image_feature, text_feature = self.compute_weiTexts(image_query, image_value, text_key, text_value, text_length, self.args)
			cmpc_loss, image_precision, text_precision = self.compute_cmpc_loss(image_embeddings, text_embeddings, labels)
		if self.CONT:
			i2t_sim, t2i_sim = self.compute_weiTexts(image_query, image_value, text_key, text_value, text_length, self.args)
			cont_loss, local_pos_avg_sim, local_neg_avg_sim = self.contrastive_loss(i2t_sim, t2i_sim, labels)
			cont_loss = cont_loss * self.args.lambda_cont
			# cont_loss = self.contrastive_loss(i2t_sim, t2i_sim, labels) * self.args.lambda_cont
		
		if cmpm_loss < 0.5:
			loss = 4*cmpc_loss + cont_loss
		else:
			loss = cmpm_loss + 4 * cmpc_loss + cont_loss
		
		return cmpm_loss, cmpc_loss, cont_loss, loss, image_precision, text_precision, pos_avg_sim, neg_avg_sim, local_pos_avg_sim, local_neg_avg_sim


class AverageMeter(object):
	"""
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py #L247-262
    """
	
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	
	def update(self, val, n=1):
		self.val = val
		self.sum += n * val
		self.count += n
		self.avg = self.sum / self.count


# 计算评价指标——top-k
def compute_topk(query_global, query, value_bank, gallery_global, gallery_key, gallery_value,
                 gallery_length, target_query, target_gallery, args, k_list=[1, 5, 10], reverse=True):  # 这里的query是images，gallery是captions
	result = []
	#sim_cosine = []
	
	query_global = F.normalize(query_global, p=2, dim=1)
	gallery_global = F.normalize(gallery_global, p=2, dim=1)
	
	# sim_cosine_global = torch.matmul(gallery_global, query_global.t())
	sim_cosine_global = torch.matmul(query_global, gallery_global.t())
	sim_cosine, _ = Loss.compute_weiTexts(query, value_bank, gallery_key, gallery_value, gallery_length, args)

	#for i in range(0, query.shape[0], 200):
	#	i2t_sim, t2i_sim = Loss.compute_weiTexts(query[i:i + 200], value_bank[i:i + 200], gallery_key, gallery_value, gallery_length, args)
	#	sim = i2t_sim
	#	sim_cosine.append(sim)
	#

	#sim_cosine = torch.cat(sim_cosine, dim=0)  # torch.Size([3074, 6156])
	sim_cosine_all1 = sim_cosine_global
	sim_cosine_all2 = 0.95*sim_cosine_global + 0.05*sim_cosine
	sim_cosine_all3 = 0.92*sim_cosine_global + 0.08*sim_cosine
	sim_cosine_all4 = 0.9*sim_cosine_global + 0.1*sim_cosine
	sim_cosine_all5 = 0.88*sim_cosine_global + 0.12*sim_cosine
	sim_cosine_all6 = 0.85*sim_cosine_global + 0.15*sim_cosine
	sim_cosine_all7 = 0.82*sim_cosine_global + 0.18*sim_cosine

	# sim_cosine_all = sim_cosine_global
	reid_sim = None
	if args.reranking:
		reid_sim = torch.matmul(query_global, query_global.t())  # torch.Size([3074, 3074])
	
	if reverse:
		result.extend(topk(sim_cosine_all1, target_query, target_gallery, k_list, dim=0, reid_sim=None))  # text to image
		result.extend(topk(sim_cosine_all4, target_query, target_gallery, k_list, dim=0, reid_sim=reid_sim))  # text to image
		result.extend(topk(sim_cosine_all2, target_query, target_gallery, k_list, dim=0, reid_sim=None))  # text to image
		result.extend(topk(sim_cosine_all3, target_query, target_gallery, k_list, dim=0, reid_sim=None))  # text to image
		result.extend(topk(sim_cosine_all4, target_query, target_gallery, k_list, dim=0, reid_sim=None))  # text to image
		result.extend(topk(sim_cosine_all5, target_query, target_gallery, k_list, dim=0, reid_sim=None))  # text to image
		result.extend(topk(sim_cosine_all6, target_query, target_gallery, k_list, dim=0, reid_sim=None))  # text to image
		result.extend(topk(sim_cosine_all7, target_query, target_gallery, k_list, dim=0, reid_sim=None))  # text to image

	return result


def topk(sim, target_gallery, target_query, k=[1, 5, 10], dim=1, reid_sim=None):
	result = []
	maxk = max(k)  # 10
	size_total = len(target_query)  # 3074  /  6126(reverse)
	
	if reid_sim is None:
		_, pred_index = sim.topk(maxk, dim, True, True)
		pred_labels = target_gallery[pred_index]
	else:
		print("reid_sim is not none")
		K = 5
		sim = sim.cpu().numpy()
		reid_sim = reid_sim.cpu().numpy()
		pred_index = np.argsort(-sim, axis=1)  # axis=1指按列排序，即对每一个text query，按照sim值从大到小对3074个image进行排列   3074*6126?
		reid_pred_index = np.argsort(-reid_sim, axis=1)
		# print("1", len(pred_index[0]), len(reid_pred_index[0]))
		
		q_knn = pred_index[:, 0:K]  # 3074*5
		g_knn = reid_pred_index[:, 0:K]  # 3074*5
		# print("2", len(q_knn[0]), len(g_knn[0]))
		
		# new_index = []
		jaccard_dist = np.zeros_like(sim)
		# sim_converse = np.zeros_like(sim)
		
		for i, qq in enumerate(q_knn):  # 3074维
			for j, gg in enumerate(g_knn):  # 3074维，  qq, gg 都是index    len(gg)=5
				jaccard_dist[i, j] = 1.0 - jaccard(qq, gg)
				# print(images[gg].size())    # torch.Size([5, 768])
				# print(torch.matmul(images[gg], captions.t()).size())
				# sim_converse[i, j] = torch.matmul(images[gg], captions.t())
		
		_, pred_index = torch.Tensor(sim + 1.7 * jaccard_dist).topk(maxk, dim, True, True)  # torch.Size([10, 6156]) (reverse)
		pred_labels = target_gallery[pred_index]
	
	if dim == 1:
		pred_labels = pred_labels.t()
	
	correct = pred_labels.eq(target_query.view(1, -1).expand_as(pred_labels))
	
	for topk in k:
		# correct_k = torch.sum(correct[:topk]).float()
		correct_k = torch.sum(correct[:topk], dim=0)
		correct_k = torch.sum(correct_k > 0).float()
		result.append(correct_k * 100 / size_total)
	
	return result


# Jaccard Distance  for RVN
def jaccard(a_list, b_list):
	return 1.0 - float(len(set(a_list) & set(b_list))) / float(len(set(a_list) | set(b_list))) * 1.0