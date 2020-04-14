import os
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from Model import NewDST
from make_dataset import *
from load_data import *
from utils.config import args
args['path'] = "save"
print("Load the DST Model...")
model = NewDST(args, 
    Vocab = Vocab, # id --> token
    slots=ALL_SLOTS,
    gating_dict=gating_dict, )
model.eval()

if __name__ == '__main__':
    # sample 1
    previous_utterances = '[A]  [U]'
    current_utterances = '[A]  [U] hi , i am looking for a train that is going to cambridge and arriving there by 20:45 , is there anything like that ?'
    previous_dict = {}

    sample = dict()
    sample['previous_utterances'] = previous_utterances
    sample['current_utterances'] = current_utterances
    sample['previous_dict'] = previous_dict
    sample['input_seq'] = get_input_seq(previous_utterances, current_utterances, previous_dict)
    sample['previous_generate_y'] = get_generate_y(sample['previous_dict'])

    result =  model.demo(sample)
    print(result)
    assert result["value"] == set(['train-destination-cambridge', 'train-arriveby-20:45'])

        
    
    
    
    
    
    
    
    
    
    
    
    
    
