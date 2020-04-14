import os
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertModel, BertTokenizer
import warnings
warnings.filterwarnings('ignore')
from make_dataset import *
from load_data import word2id, Vocab, USE_CUDA, train_data, dev_data, test_data, LoadData
from utils.loss_drawer import LossSaver
from Model import NewDST
from utils.config import args

# 1. hyparameters
#args['learning_rate'] = 0.001
#print(args)
clip = int(args['clip'])
epoch_num = args["epoch_num"]
args['batch_size'] = 16
dev_batch = 1
early_stop = args['earlyStop']

#args['path'] = False#"save"
args['evalp'] = 1
print("Training {0} Epoches".format(epoch_num))
print("Load data.. \n")

train = LoadData(train_data, word2id, args['batch_size'], args['use_weighted_sampler'])
dev   = LoadData(dev_data, word2id, dev_batch)
test  = LoadData(test_data, word2id, dev_batch)
num_total_steps = int(len(train_data) / args['batch_size'] * epoch_num)
print('number of samples:', len(train_data), len(train))
print("torch.cuda.is_available():", USE_CUDA)
# 2. train
avg_best, cnt, acc = 0.0, 0, 0.0
# Instantiate the model
print("Load the DST Model...")
model = NewDST(args, 
    Vocab = Vocab, # id --> token
    slots=ALL_SLOTS,
    gating_dict=gating_dict, 
    num_total_steps=num_total_steps,)
#torch.manual_seed(1234)
if USE_CUDA:
    torch.cuda.set_device(0)
    #torch.cuda.manual_seed(1234)
#for n, p in model.named_parameters():
    #print(n)
global_loss_list= []

for epoch in range(epoch_num):
    print("Epoch:{}".format(epoch))  
    # Run the train function
    pbar = tqdm(enumerate(train),total=len(train)) # 13746
    for i, data in pbar:
        model.train()
        out = model.Train(data, reset=(i==0))
        model.optimize(clip) # model.optimize.step()  model.scheduler.step()
        losses = model.print_loss()
        #lr = model.optimizer.state_dict()['param_groups'][0]['lr']
        #global_loss_list.append(losses+(lr,))
        pbar.set_description('L:{:.4f},Lptr:{:.4f},Lopr:{:.4f},Ldom:{:.4f}, Lcfm:{:.4f}'.format(*losses))
        
    #if args['genSample']==1:
        #LossSaver(global_loss_list)
    #if((epoch+1) % int(args['evalp']) == 0):
    with torch.no_grad():
        acc = model.evaluate(dev, avg_best, ALL_SLOTS, early_stop)
    model.zero_grad()
    if(acc >= avg_best):
        avg_best = acc
        cnt = 0
        best_model = model
    else:
        cnt += 1
    if (cnt == args["patience"] or (acc == 1.0 and early_stop==None)): 
        print("Ran out of patient, early stop...")  
        break 
print("Finished..")


