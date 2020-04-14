import os
import pickle
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from make_dataset import special_tokens, EXPERIMENT_DOMAINS, UNK_token, PAD_token, EOS_token, SEP_token, A_token, U_token, CLS_token, NULL_token, path, gating_dict, ontology, ALL_SLOTS
from make_dataset import bert_type, bert_vocab_path, tokenizer, bert_model_path, version
# 1. global varibales
# special_token index
UNK_token_id = tokenizer.convert_tokens_to_ids(UNK_token)
SEP_token_id = tokenizer.convert_tokens_to_ids(SEP_token)
PAD_token_id = tokenizer.convert_tokens_to_ids(PAD_token)
CLS_token_id = tokenizer.convert_tokens_to_ids(CLS_token)
EOS_token_id = tokenizer.convert_tokens_to_ids(EOS_token)
A_token_id = tokenizer.convert_tokens_to_ids(A_token)
U_token_id = tokenizer.convert_tokens_to_ids(U_token)
NULL_token_id = tokenizer.convert_tokens_to_ids(NULL_token)



# load vocab(token2id)
use_BertVocab = True 
if use_BertVocab:
    Vocab = tokenizer.ids_to_tokens
else:
    Vocab = torch.load('Vocab_dic.dict')

word2id = {i:v for (v,i) in Vocab.items()} 
# Load dataset
with open('train{}.pkl'.format(version), 'rb') as f:
    mysave_train = pickle.load(f)
train_data, train_max_input, train_max_value = mysave_train['data'], mysave_train['max_input'], mysave_train['max_value']

with open('dev{}.pkl'.format(version), 'rb') as f:
    mysave_dev = pickle.load(f)
dev_data, dev_max_input, dev_max_value = mysave_dev['data'], mysave_dev['max_input'], mysave_dev['max_value']
with open('test{}.pkl'.format(version), 'rb') as f:
    mysave_test = pickle.load(f)
test_data, test_max_input, test_max_value = mysave_test['data'], mysave_test['max_input'], mysave_test['max_value']



# 2. Hyparamaters
from utils.config import USE_CUDA, args
batch_size = args['batch_size']
#max_input_seq =  256
if USE_CUDA:
    torch.set_default_tensor_type('torch.cuda.FloatTensor') 

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, word2id, sequicity=0):
        """source"""
        self.input_seq = data_info['input_seq']
        self.previous_generate_y = data_info['previous_generate_y']
        self.num_total_seqs = len(self.input_seq)
        """label"""
        try:
            self.ID = data_info['ID']
            self.turn_id = data_info['turn_id']
            self.turn_belief = data_info['turn_belief']
            self.generate_y = data_info['generate_y']
            self.gating_label = data_info['gating_label']
            self.domain_focus = data_info['domain_focus']
            #self.previous_utterances = data_info['previous_utterances']
            #self.current_utterances = data_info['current_utterances']
            #self.domains = data_info['domains'] # 没有domains
            #self.turn_domain = data_info['turn_domain']
            #self.story = data_info['story']
            #self.previous_belief = data_info['previous_belief']
        except KeyError:
            self.ID = [""] * self.num_total_seqs
            self.turn_id = [0] * self.num_total_seqs
            self.turn_belief = [[]] * self.num_total_seqs
            self.generate_y = [[NULL_token] * len(ALL_SLOTS)] * self.num_total_seqs
            self.gating_label = [[0]*len(ALL_SLOTS)] * self.num_total_seqs
            self.domain_focus = [[0]*len(EXPERIMENT_DOMAINS)] * self.num_total_seqs

        self.word2id = word2id
    
    def __getitem__(self, index): 
        """Returns one data pair (source and target)."""
        ID = self.ID[index]
        #turn_domain = self.turn_domain[index]  
        turn_id = self.turn_id[index]
        turn_belief = self.turn_belief[index]
        #previous_utterances = self.previous_utterances[index]
        #current_utterances = self.current_utterances[index]
        #turn_belief = self.turn_belief[index]
        previous_generate_y = self.previous_generate_y[index] # 用于carryover，所以不索引化

        generate_y = self.generate_y[index]
        
        gating_label = self.gating_label[index]
        domain_focus = self.domain_focus[index]
        
        #story = self.story[index]
        #previous_belief = self.previous_belief[index]
        input_seq = self.input_seq[index]
        previous_generate_y_idx = self.token2idx_slot(previous_generate_y, self.word2id)
        generate_y_idx = self.token2idx_slot(generate_y, self.word2id)

        input_seq_idx = self.token2idx_seq(input_seq, self.word2id)        
        
        item_info = {
            "ID":ID, 
            "turn_id":turn_id, 
            "turn_belief":turn_belief,
            "gating_label":gating_label,
            "input_seq":input_seq, 
            "input_seq_idx":input_seq_idx, 
            "previous_generate_y":previous_generate_y, # carryover
            "previous_generate_y_idx":previous_generate_y_idx,
            "domain_focus": domain_focus, 
            "generate_y":generate_y,
            "generate_y_idx":generate_y_idx,
            }
        return item_info

    def __len__(self):
        return self.num_total_seqs
    
    def token2idx_seq(self, sequence, word2idx):
        """Converts words to ids."""
        idx_seq = [word2idx[word] if word in word2idx else UNK_token_id for word in sequence]
        idx_seq = torch.Tensor(idx_seq)
        return idx_seq

    def token2idx_slot(self, sequence, word2idx):
        """Converts words to ids."""
        idx_seq = []
        for value in sequence:
            # none:idx 212;EOS:idx 2;PAD:idx 1.
            # [EOS_token]:decoder ending flag
            try:
                v = [word2idx[word] if word in word2idx else UNK_token_id for word in tokenizer.tokenize(value)] + [EOS_token_id]
            except:
                print("出现错误:\n", sequence)
                print("v:", value)
            idx_seq.append(v) 
        # story = torch.Tensor(story)
        return idx_seq

    def token2idx_domain(self, turn_domain):
        #domains = {"attraction":0, "hotel":1, "restaurant":2, "taxi":3, "train":4, "hospital":5, "bus":6, "police":7}
        domains = {"attraction":0, "hotel":1, "restaurant":2, "taxi":3, "train":4}
        return domains[turn_domain]


# Do PADDING for input_seq & output_valueSeq
def collate_fn(data):
    def merge(sequences):
        #merge from batch * sent_len to batch * max_len 
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        #max_len = max_input_seq # forcing
        padded_seqs = torch.zeros(len(sequences), max_len).long() # pad是全0
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach() #torch.tensor(padded_seqs)
        return padded_seqs, lengths

    def merge_multi_response(sequences):
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        max_len = max([max(l) for l in lengths]) 
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [PAD_token_id] * (max_len-len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        #print(padded_seqs)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths
  
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['input_seq_idx']), reverse=True) 
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['input_seq_idx'])
    y_seqs, y_lengths = merge_multi_response(item_info["generate_y_idx"])

    gating_label = torch.tensor(item_info["gating_label"])
    domain_focus = torch.tensor(item_info["domain_focus"])

    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        y_seqs = y_seqs.cuda()
        y_lengths = y_lengths.cuda()
        gating_label = gating_label.cuda()
        domain_focus = domain_focus.cuda()
        

    item_info["input_seq_idx"] = src_seqs
    item_info["input_len"] = src_lengths # true input-seq length

    item_info["gating_label"] = gating_label
    item_info["domain_focus"] = domain_focus
    
    item_info["generate_y_idx"] = y_seqs
    item_info["y_lengths"] = y_lengths # true value-seq length
    return item_info

def LoadData(data, word2id, batch_size, use_weighted_sample=False):
    data_info = dict()
    data_keys = data[0].keys()
    for k in data_keys:
        data_info[k] = []
    for pair in data:
        for k in data_keys:
            data_info[k].append(pair[k])
    dataset = Dataset(data_info, word2id, sequicity=0)

    if use_weighted_sample == True:
        weights = [1 if data["gating_label"] == [0]*len(ALL_SLOTS) else 9 for data in dataset]
        from torch.utils.data.sampler import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    else:
        sampler = None
   
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, sampler=sampler, collate_fn=collate_fn)
    #data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loader


if __name__ == '__main__':
    print('Vocab Length:', len(Vocab))
    data = test_data
    
    data_info = dict()
    data_keys = data[0].keys()
    for k in data_keys:
        data_info[k] = []
    for pair in data:
        for k in data_keys:
            data_info[k].append(pair[k])
    dataset = Dataset(data_info, word2id, sequicity=0)
    weights = [1 if data["gating_label"] == [0]*len(ALL_SLOTS) else 2 for data in dataset]

    from torch.utils.data.sampler import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=4,collate_fn=collate_fn,sampler=sampler)
    for datas in dataloader:
        print(datas)



