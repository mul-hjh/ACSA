import torch
from torch import nn
from pytorch_transformers import BertModel, BertConfig, BertTokenizer


class TextNet(nn.Module):
	def __init__(self, args):
		super(TextNet, self).__init__()
		self.image_model = args.image_model
		
		self.tokenizer = BertTokenizer.from_pretrained('/home/junhua/pretrained_model/bert-base-uncased/bert-base-uncased-vocab.txt')  # 通过词典导入分词器
		modelConfig = BertConfig.from_pretrained('/home/junhua/pretrained_model/bert-base-uncased/bert_config.json')  # 导入配置文件
		self.textExtractor = BertModel.from_pretrained('/home/junhua/pretrained_model/bert-base-uncased/pytorch_model.bin', config=modelConfig)  # 使用预训练好的BERT
		
	
	def pre_process(self, texts, length):
		tokens, segments, input_masks = [], [], []
		text_length = []
		
		for text in texts:
			text = '[CLS] ' + text + ' [SEP]'
			tokenized_text = self.tokenizer.tokenize(text)
			indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
			if len(indexed_tokens) > length:
				indexed_tokens = indexed_tokens[:length]
				
			tokens.append(indexed_tokens)
			segments.append([0] * len(indexed_tokens))
			input_masks.append([1] * len(indexed_tokens))
		
		# max_len = max([len(single) for single in tokens])
		
		for j in range(len(tokens)):
			padding = [0] * (length - len(tokens[j]))    # 这一步严重增加的计算复杂度
			text_length.append(len(tokens[j])+3)
			tokens[j] += padding
			segments[j] += padding
			input_masks[j] += padding
		
		tokens = torch.tensor(tokens)
		segments = torch.tensor(segments)
		input_masks = torch.tensor(input_masks)
		text_length = torch.tensor(text_length)
		
		return tokens, segments, input_masks, text_length
	
	def forward(self, tokens, segments, input_masks):    # output[0]是word embeddings，output[1]是sentence embeddings
		output = self.textExtractor(tokens, token_type_ids=segments, attention_mask=input_masks)    # len=2, output[0]: [bs, 100, d]    output[1]: [bs, d]
		# text_embeddings = output[0][:, 0, :]
		text_embeddings = output[0]    # sentence: torch.Size([bs, 100, d])     sub-sentence: torch.Size([2*bs, 100, d])
		
		return text_embeddings