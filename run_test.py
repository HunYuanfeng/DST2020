# -*- coding: utf-8 -*-
"""
测试脚本: 多种模式
1. 基本模型: 使用slot gate, max-res-len=10

2.模式1: 使用domain-interest修正slot gate

3.模式2: 使用GT slot gate

"""
import os
import pickle
from tqdm import tqdm
from pytorch_transformers import BertModel, BertTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.utils.data as data

from make_dataset import *
from make_dataset import bert_type, bert_vocab_path, tokenizer, bert_model_path
from load_data import word2id, Vocab, USE_CUDA, max_input_seq, train_data, dev_data, test_data, LoadData

from Model import NewDST
# 1. 超参数
from utils.config import args
args['learning_rate'] = 0.001
#print(args)
clip = int(args['clip'])
epoch_num = 1
args['batch_size'] = 16
dev_batch = 4
early_stop = args['earlyStop']

args['path'] = "save"
#args['domain_as_task'] = False
args['evalp'] = 1

print("Load data.. \n")
#train = LoadData(train_data, word2id, args['batch_size'], args['use_weighted_sampler'])
dev   = LoadData(dev_data, word2id, dev_batch)
test  = LoadData(test_data, word2id, dev_batch)

print("torch.cuda.is_available():", USE_CUDA)
# 实例化模型
print("Load the DST Model...")
model = NewDST(args, 
    Vocab = Vocab, # id --> token
    slots=ALL_SLOTS,
    gating_dict=gating_dict, )


avg_best = 1e9
if args['rundev']:
    print("\ndev Set ...")
    acc_test = model.evaluate(dev, avg_best, ALL_SLOTS)
if args['runtest']:
    print("\nTest Set ...")
    acc_test = model.evaluate(test, avg_best, ALL_SLOTS)


