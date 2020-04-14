import os
import json
import pickle
from utils.fix_label import fix_general_label_error
# fix_label herited from TRADE except 14 new-added pairs
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import torch
from utils.config import args
bert_type = 'bert-base-uncased'

change_unused = True # 当对SOM-DST的特殊情况，把原始词表里的[unusedX]替换为special tokens

if not change_unused:
    bert_vocab_path=os.path.join(os.getcwd(), "transformers") + '/%s-vocab.txt' % bert_type
else:
    bert_vocab_path=os.path.join(os.getcwd(), "transformers") + '/%s-vocab-special.txt' % bert_type
bert_model_path=os.path.join(os.getcwd(), "transformers") + '/%s.model' % bert_type

#print("bert_vocab_path:", bert_vocab_path)
version = args['dataset'].split("multiwoz")[-1] # '2.1'

print('Using dataset Multiwoz {0}'.format(version))
path = 'data{}'.format(version)
file_train = os.path.join(path, 'train_dials.json')
file_dev = os.path.join(path, 'dev_dials.json')
file_test = os.path.join(path, 'test_dials.json')
folder_name = 'save/'
if not os.path.exists(folder_name): 
    os.makedirs(folder_name)
    
EXPERIMENT_DOMAINS = ('attraction', 'hotel', 'restaurant', 'taxi', 'train')
cl_dict={'pricerange':'price range','leaveat':'leave at','arriveby':'arrive by'}

file_ontology = os.path.join(path, 'multi-woz/ontology.json')
ontology = json.load(open(file_ontology, 'r'))
if version == '2.1':
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    ALL_SLOTS_ = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    #ALL_SLOTS = [(s.split('-')[0] + '-' + s.split('-')[2]) for s in ALL_SLOTS_]
    ALL_SLOTS = [(s.split('-')[0] + '-' + s.split('-')[2]) if s.split('-')[1]=='semi' else (s.split('-')[0] + '-' + s.split('-')[1] + " "+ s.split('-')[2]) for s in ALL_SLOTS_]
else:
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    ALL_SLOTS = [k.lower() for k in ontology_domains.keys()]

ALL_SLOTS = sorted([s.split("-") for s in ALL_SLOTS])
ALL_SLOTS = [[s[0], cl_dict.get(s[1], s[1])] for s in ALL_SLOTS]
ALL_SLOTS = ["-".join(s) for s in ALL_SLOTS]
#print('ALL_SLOTS:\n', ALL_SLOTS) # 定序列表。domain按attraction,hotel,restaurant,taxt,train排列


ALLnested = {d:[] for d in EXPERIMENT_DOMAINS}
for d in EXPERIMENT_DOMAINS:
    for ds in ALL_SLOTS:
        if ds.split("-")[0] == d: ALLnested[d].append(ds.split("-")[1])
        
ALL_ds_nb = {d:len(slots) for d, slots in ALLnested.items()}
# {'attraction': 3, 'hotel': 10, 'restaurant': 7, 'taxi': 4, 'train': 6}

# special token to added
PAD_token = "[PAD]" # index = 0
UNK_token = "[UNK]" # 
CLS_token = "[CLS]"
SEP_token = "[SEP]"
# 在bert_vocab_path下，已经预先做了如下改变:
DOM_token = "[DOM]"
SLOT_token = "[SLOT]"
EOS_token = "[EOS]" # 原[unused2]  
NULL_token = "[NULL]" # 原[unused3] 
A_token = "[A]" # 原[unused4]  
U_token = "[U]" # 原[unused5] 
special_tokens = [EOS_token, DOM_token, SLOT_token, NULL_token, A_token, U_token]

tokenizer = BertTokenizer.from_pretrained(bert_vocab_path)
tokenizer.additional_special_tokens = special_tokens # 使得分词时对它们视为一个token

gating_dict = {"carryover":0, "confirm":1, "update":2}

sub_gating_dict = {"yes":0, "no":1, NULL_token:2, "do not care":3} # 与4个value一一对应
sub_gating_dict_verse = {i:v for (v,i) in sub_gating_dict.items()} 
domain_gating_dict = {"carryover":0, "focus":1}


# domain-slot token也做了对应的replace
# 基于此构建全局字典(spe-tok & word & domain-slot-svalue)

use2turn = False
"""
def fix(dic):
    for key in list(dic.keys()):
        if dic[key] == 'none':
            del dic[key]
        elif dic[key] == 'dontcare':
            dic[key] = 'do not care'
"""
# 小工具函数:将value中的dontcare都换成do not care, none都换成[NULL]
def fix(v):
    if v == 'dontcare' :
        return 'do not care'
    if v == 'none':
        return NULL_token
    else:
        return str(v)

# 有4个相似的变量:
#turn_belief_dict: 数据集本身的格式, 是一个s:v字典
#turn_belief_list: d-s-v的字符串
#generate_y: v的字符串
#bs:input-seq的内容
def _convert_bs(turn_belief_dict):
    bs = ""
    for dom, slots in ALLnested.items():
        bs += (' [DOM] ' + dom +  ' : ')
        for slot_name in slots:
            slot_name = cl_dict.get(slot_name, slot_name)
            v = turn_belief_dict.get("-".join([dom, slot_name]), "[NULL]")
            bs += (' [SLOT] ' + slot_name +  ' - ' + v)
    # print(b.count("-"), b.count(":")) # 30, 5
    #b_ = tokenizer.encode(bs)
    #b = tokenizer.convert_ids_to_tokens(b_)
    return bs

def get_input_seq(previous_utterances, current_utterances, previous_dict):
    previous_belief = _convert_bs(previous_dict).strip()
    story = previous_utterances + " " + SEP_token + " " + current_utterances + " " + SEP_token
    input_seq = [CLS_token] + tokenizer.tokenize(story + " " + previous_belief) + [EOS_token]
    return input_seq


def get_generate_y(turn_belief_dict):
    if turn_belief_dict:
        generate_y = []
        for slot in ALL_SLOTS:
            if slot in turn_belief_dict.keys(): 
                value = turn_belief_dict[slot]
                value = fix(value)
                value_seq = tokenizer.tokenize(turn_belief_dict[slot])
            else:
                value = NULL_token
            generate_y.append(value)
    else:
        generate_y = [NULL_token] * len(ALL_SLOTS)
    return generate_y
        
def get_belief_dict(generate_y):
    turn_belief_dict = {s:v for s, v in zip(ALL_SLOTS, generate_y) if v != '[NULL]'}
    return turn_belief_dict   

def get_domain_focus(generate_y, gating_label):
    slot_4_tuple = [tuple(s.split("-"))+(v, sg,) for s, sg, v in zip(ALL_SLOTS, gating_label, generate_y)]
    dom_gate = list(set([each[0] for each in list(filter(lambda e: e[-1] != gating_dict["carryover"], slot_4_tuple))]))
    return [int(d in dom_gate) for d in EXPERIMENT_DOMAINS]

def get_data(file_data):
    print(("Reading from {}".format(file_data)))
    with open(file_data) as f:
        dials = json.load(f) # len(dials) = 8420
        
    data = [] # data是全部dialogue的全部turn，逐turn封装的训练数据。
    max_value_len, max_input_len, max_value, max_input = 0, 0, '', ''
    dials = tqdm(enumerate(dials),total=len(dials)) # 13746
    for i, dial_dict in dials:
    #for dial_dict in dials:
        '''
        for domain in dial_dict["domains"]: # 放在外循环内!
            if domain not in EXPERIMENT_DOMAINS:
                continue
        '''
        if not set(dial_dict["domains"]) < set(EXPERIMENT_DOMAINS):
            continue
        #dialog_history = ''
        previous_utterances = A_token + " " + U_token 
        previous_generate_y = [NULL_token] * len(ALL_SLOTS)
        previous_belief = _convert_bs({}).strip()
        previous_gating_label = [gating_dict["carryover"]] * len(ALL_SLOTS)
        # 整个对话的开始之前的state B_0，has only NULL as the value of all slots
        # 所以B_1对应的gate是Carryover
        #previous_generate_y = ['' for s in ALL_SLOTS]

        for ti, turn in enumerate(dial_dict["dialogue"]):
            # 1 基本
            turn_domain = turn["domain"]
            turn_id = turn["turn_idx"]

            turn_belief_dict = fix_general_label_error(turn["belief_state"], False, ALL_SLOTS) # 修正错误,取value

            # Generate domain-dependent slot list(target)
            if turn_belief_dict:
                turn_belief_list = [str(k)+'-'+fix(v) for k, v in turn_belief_dict.items() if v != "none"]  
                turn_belief_dict = {k: fix(v) for k, v in turn_belief_dict.items()}
            else:
                turn_belief_list = []
            # 2 生成自定义label
            # 2.1 生成generate_y:按ALL_SLOTS排列的value列表。
            if turn_belief_dict:
                generate_y = []
                for slot in ALL_SLOTS:
                    if slot in turn_belief_dict.keys(): 
                        value = turn_belief_dict[slot]
                        value = fix(value)
                        value_seq = tokenizer.tokenize(turn_belief_dict[slot])
                        if max_value_len < len(value_seq):
                            max_value_len = len(value_seq)       
                            max_value = value_seq
                    else:
                        value = NULL_token
                    generate_y.append(value)
                #print('generate_y:\n',generate_y)
            else:
                generate_y = [NULL_token] * len(ALL_SLOTS)
            # 2.2 用相邻轮的generate_y生成gate_label:按ALL_SLOTS排列的operation列表
            gating_label = []
            for sidx in range(len(ALL_SLOTS)):
                p_v = previous_generate_y[sidx]
                n_v = generate_y[sidx]
                if n_v == p_v:
                    operation = gating_dict["carryover"]
                # yes/no/dontcare
                elif n_v in list(sub_gating_dict.keys()):
                    operation = gating_dict["confirm"]
                else:
                    operation = gating_dict["update"] 
                gating_label.append(operation)
            #print(gating_label)
            
    
            # 2.3 生成input_seq & previous_belief：
            #print('previous_utterances:\n', previous_utterances)
            #print('current_utterances:\n', current_utterances)
            # input = CLS_token + previous_utterances + SEP_token + current_utterances + previous_belief
            
            #tokenizer or split:用tokenizer
            
            '''
            三个seq:
               story:包括D_t-1和D_t,以及两个SEP
               previous_belief:包括B_t-1
               input_seq: story和previous_belief拼接，同时前加一个CLS，尾加一个SEP
            '''
            # 2.3.1 story
            current_utterances = A_token + " " + turn["system_transcript"] + " " + U_token + " " + turn["transcript"]
            current_utterances = current_utterances.strip()
            story = previous_utterances + " " + SEP_token + " " + current_utterances + " " + SEP_token
            #story = tokenizer.tokenize(previous_utterances) + [SEP_token] + tokenizer.tokenize(current_utterances) + [SEP_token] 

            # 2.3.2 previous_belief(应该是30个value都放入input吧，包括[NULL] value)
            current_belief = _convert_bs(turn_belief_dict).strip()

            # 2.3.3 input_seq
            input_seq = [CLS_token] + tokenizer.tokenize(story + " " + previous_belief) + [EOS_token]

            if max_input_len < len(input_seq):
                max_input_len = len(input_seq)
                max_input = input_seq
            
            # 2.3.4 三元组
            #SEP_indices = tuple([i for i, x in enumerate(input_seq) if x == SEP_token])
            #assert len(SEP_indices) == 2
           # 2.4 生成domain_focus：0表示领域完全地不关注；1表示领域中存在至少一个slot是非carryover的
            # domain_focus
            domain_focus = get_domain_focus(generate_y, gating_label)
            
            if use2turn:
                pre_domain_focus = get_domain_focus(previous_generate_y, previous_gating_label)
                domain_focus = list(map(lambda x:(x[0]|x[1]) ,zip(domain_focus, pre_domain_focus)))


            # 3. 用一个容器装起来：
            data_detail = {
            "ID":dial_dict["dialogue_idx"], 
            "domains":dial_dict["domains"], # 这个turn所属的对话共涉及哪些领域
            "turn_domain":turn_domain, 
            "turn_id":turn_id, 
            "turn_belief":turn_belief_list,
            'previous_generate_y':previous_generate_y,
            'generate_y':generate_y,
            "gating_label":gating_label,
            "previous_utterances":previous_utterances,
            "current_utterances":current_utterances,
            'domain_focus': domain_focus,
            #'story':story, # D_{t-1}+Dt. 仅当前轮和上一轮的对话原文 (str)
            #'previous_belief':previous_belief, # B_t (str)
            'input_seq':input_seq, # D_{t-1}+Dt+Bt, 已经分词 (list)
            #'SEP_indices':SEP_indices, # Dt-1,Dt,Bt-1的结束位置 (tuple)
            }
            data.append(data_detail)
            
            previous_utterances, previous_generate_y, previous_belief, previous_gating_label = current_utterances, generate_y, current_belief, gating_label
            #previous_turn_domain = turn_domain]
    #print('max_value:\n', max_value) #  london liverpool street=23(char级别),3(tk级别)
    #print(max_value_len)
        #break
    return data, (max_input_len, max_input), (max_value_len, max_value)


# 2. 构建自定义数据
'''
train_data, train_max_input, train_max_value = get_data(file_train) # 56668;204;3

# 为了不浪费重新处理的时间，将它们存入本地
mysave_train = {'data':train_data, 'max_input':train_max_input, 'max_value':train_max_value}    
with open('mysave_train2.1.pkl', 'wb') as f:
    pickle.dump(mysave_train, f)
'''

# 用一个函数跑完三个数据集
def save_data(file_data, file_name):
    data, max_input, max_value = get_data(file_data)
    mysave = {'data':data, 'max_input':max_input, 'max_value':max_value} 
    with open(file_name, 'wb') as f:
        pickle.dump(mysave, f)    
    print('Succesfully trans {0}.'.format(file_data))
    
    
if __name__ == '__main__':
    #version = '2.1'
    print('Domains num:{0}; Slots num:{1}'.format(len(EXPERIMENT_DOMAINS), len(ALL_SLOTS)))#5 30
    print("Start make datasets...")
    save_data(file_train, 'train{}.pkl'.format(version))
    save_data(file_dev, 'dev{}.pkl'.format(version))
    save_data(file_test, 'test{}.pkl'.format(version))
    

 
    
    
    
    
    
    
    