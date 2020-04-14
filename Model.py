import re
import os
import json
import pickle
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
#from torch.autograd import Variable
from torch.optim import lr_scheduler
#from torch import optim
import torch.nn.functional as F
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import nltk
from make_dataset import bert_model_path, tokenizer, EXPERIMENT_DOMAINS, ALL_ds_nb, sub_gating_dict, sub_gating_dict_verse, DOM_token, SLOT_token, NULL_token, CLS_token, EOS_token, SEP_token
from load_data import *
from utils.criterion import FocalLoss
#from utils.attention import sequence_mask, global_attention
from utils.config import args, USE_CUDA

import warnings
warnings.filterwarnings('ignore')

bmodel = BertModel.from_pretrained(bert_model_path)
bmodel.eval()

global_loss_list = []
if USE_CUDA:
    bmodel.to('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

import time
#获取今天的字符串
def getToday():
    return time.strftime("%Y-%m-%d",time.localtime(time.time()))

def warmup_linear(x, warmup):
    if x < warmup:
        return x / warmup
    return 1.0 - x

class NewDST(nn.Module):
    ''' 
    构造f方法输入：
    从前几步继承来的slot、gate、vocab number，词表Vocab
    外加三个超参数：词向量维度1024, 隐藏层维度768, 词表长度, 学习率0.001，dropout系数0.1
    其他的path表示DST模型过去训练的权重值，可以选择继续训练
    '''
    def __init__(self, args, Vocab, slots, gating_dict, num_total_steps=0):
        super(NewDST, self).__init__()
        self.name = "NewDST" 
        self.id2word = Vocab # dict (index --> token)  
        self.vocab_size = len(self.id2word)
        self.slots = slots
        self.gating_dict = gating_dict # type --> id
        self.nb_gate = len(gating_dict) # 4
        self.batch_size = args['batch_size']
        self.hidden_size = args['hidden_size']   
        self.dropout = args['dropout']
        
        # Loss function (reduce='mean')
        self.loss_func_ptr = nn.CrossEntropyLoss(ignore_index=0) # PAD_token_id: 0
        #self.loss_func_ptr = FocalLoss(class_num=self.vocab_size, ignore_index=0)
        self.loss_func_opr = FocalLoss(class_num=self.nb_gate) # [1, 170, 29]
        self.loss_func_dom = FocalLoss(class_num=2) #0.15/ 0.2
        self.loss_func_cfm = FocalLoss(class_num=len(sub_gating_dict))

        # 实例化encoder、decoder
        self.encoder = EncoderBERT(self.hidden_size, self.slots, self.gating_dict)
        Word_Embedding = self.encoder.transformer.embeddings.word_embeddings # nn.Embedding对象, 其weight是[30522, 768]
        
        self.decoder = Generator(self.id2word, self.vocab_size, self.hidden_size, self.dropout, self.slots, self.gating_dict, Word_Embedding) 
        
        # 参考save_model方法。这里是把预训练模型导入进来
        # 方法: model.load_state_dict((torch.load(dir)).state_dict())
        # cpu/gpu相互转换函数map_location:https://www.cnblogs.com/xiaodai0/p/10413711.html
        path = args["path"]
        if path:
            model_file = path + '/checkpoint.pt'
            print("MODEL {} is Loaded...".format(model_file))
            if USE_CUDA:
                trained_state_dict = torch.load(model_file)
            else:
                trained_state_dict = torch.load(model_file,lambda storage, loc: storage)
            self.load_state_dict(trained_state_dict)
        else:
             print("Don't use Pretrained-MODEL.")
  
        self.lr = (1e-4, 4e-5,)
        self.num_total_steps = num_total_steps
        
        decoder_params = self.decoder.parameters()
        #for value in decoder_params:
            #value.requires_grad = False
        encoder_params = list(filter(lambda p: id(p) not in list(map(id, decoder_params)), self.parameters()))
        self.optimizer = AdamW([
                        {'params': decoder_params, 'lr': self.lr[0]},
                        {'params': encoder_params, 'lr': self.lr[1]}],)
        warmup_prop = args['warmup_prop'] # default=0.1
        warmup_steps = warmup_prop * self.num_total_steps
        # 跨epoch学习率控制类:当1个epoch指标还没提升时，学习率下降为初始的0.5倍，最多下降到1e-5
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                        num_warmup_steps=warmup_steps, num_training_steps=self.num_total_steps)
        
        self.reset()
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def optimize(self, clip):
        self.loss_grad.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)# clip: max_grad_norm
        
        self.optimizer.step()
        self.scheduler.step()
        
    def save_model(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.state_dict(), directory + '/checkpoint.pt')
        
    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_dom, self.loss_cfm = 0, 1, 0, 0, 0, 0
        
    def print_loss(self):    
        print_loss_avg = self.loss.item() / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_cfm = self.loss_cfm / self.print_every
        print_loss_gate = self.loss_gate / self.print_every
        print_loss_domain = self.loss_dom / self.print_every
        self.print_every += 1 
           
        return (print_loss_avg, print_loss_ptr, print_loss_gate, print_loss_domain, print_loss_cfm, )
        #return 'L:{:.4f},Lptr:{:.4f},Lopr:{:.4f},Ldom:{:.4f}'.format(print_loss_avg,print_loss_ptr,print_loss_gate,print_loss_domain)

    def encode_and_decode(self, data, use_teacher_forcing, classifiers_pred=None):
        """input sources"""
        input_seq = data['input_seq_idx']
        input_len = data["input_len"]
        batch_size, max_input_seq = input_seq.size()
        previous_gold_y = data['previous_generate_y']
        """label"""
        gold_y_idx = data['generate_y_idx']
        gold_y = data['generate_y'] 
        GoldGates = data["gating_label"]
        GoldDoms = data["domain_focus"]

        SEP_indices = torch.zeros(batch_size, 2)
        for bi in range(batch_size):
            SEP_indices[bi] = torch.tensor([i for i, x in enumerate(input_seq[bi]) if x == SEP_token_id])
            #assert len(SEP_indices) == 2
        SEP_indices = SEP_indices.int()
        
        # encode
        encoded_out = self.encoder(input_seq, input_len, max_input_seq, SEP_indices)
        encoded_outputs, X_hdd, SLOT_hdd_all, predict_gates, predict_dom = encoded_out
        encoded_cfms = self.encoder.encode_confirm_seq()
        
        
        # prepare the slot gate result(predict_gates --> PredictGates) & get the refines
        all_slots_refine = torch.zeros(batch_size, len(self.slots)) # as a func return

        # assert not args['testing_gate_mode'] < args['training_gate_mode']
        # if args['training_gate_mode']=0, args['testing_gate_mode']=0 or2
        # if args['training_gate_mode']=1, args['testing_gate_mode'] can only =2
        PredictGates = torch.zeros(batch_size, len(self.slots))
        PredictInts = torch.zeros(batch_size, len(EXPERIMENT_DOMAINS))
        if self.decoder.training:
            # train stage using groundtruth gate all the time
            PredictGates = GoldGates
            for bi in range(batch_size):
                gating_pred = PredictGates[bi, :]
                focus_dom_label_ = GoldDoms[bi]
                slots_refine = self.get_slots_refine(focus_dom_label_)
                all_slots_refine[bi, :] = slots_refine

        else: 
            # eval stage have 4 types of 'testing_gate_mode'
            for bi in range(batch_size):
                gating_pred = torch.argmax(predict_gates.transpose(0, 1)[bi], dim=1) # dim (int) – the dimension to reduce.            
                
                focus_dom_label_ = torch.argmax(predict_dom.transpose(0, 1)[bi], dim=1)
                if args['testing_gate_mode'] == 3: # use groundtruth dom-refined tensor
                    focus_dom_label_ = GoldDoms[bi]          

                slots_refine = self.get_slots_refine(focus_dom_label_)
                all_slots_refine[bi, :] = slots_refine
                gating_pred_w_d = list(map(lambda x:x[0] * x[1], zip(gating_pred, slots_refine)))
                gating_pred_w_d = torch.tensor(gating_pred_w_d)
                #gating_pred_w_d = self._refine(predict_gates.transpose(0, 1)[bi], slots_refine)
                
                if args['training_gate_mode'] == 0 and args['testing_gate_mode'] == 0: # Default
                    gates_opr = gating_pred
                elif args['testing_gate_mode'] == 1: # groundtruth slot gate
                    gates_opr = data["gating_label"][bi, :]
                else: # use dom-refined slot gate"
                    gates_opr = gating_pred_w_d
                    
                PredictGates[bi, :] = gates_opr
                PredictInts[bi, :]  = focus_dom_label_
                
                # whether do classfication evaluation
                if classifiers_pred:
                    all_dom_prediction, all_slot_prediction, all_slot_prediction_w, all_cfm_prediction = classifiers_pred
                    domL_true = GoldDoms[bi].data.tolist()
                    domL_pred = focus_dom_label_.data.tolist()
                    all_dom_prediction['y_true'] += domL_true
                    all_dom_prediction['y_pred'] += domL_pred
                    all_slot_prediction['y_true'] += GoldGates[bi].data.tolist()
                    all_slot_prediction['y_pred'] += gating_pred.data.tolist()
                    all_slot_prediction_w['y_true'] += GoldGates[bi].data.tolist()
                    all_slot_prediction_w['y_pred'] += gating_pred_w_d.data.tolist()                
                    if args["genSample"]:
                        if not torch.equal(GoldDoms[bi], focus_dom_label_):
                            tosave_d = "Dialog:{0},turn:{1}\nTrue:{2}\nPred:{3}\n".format(data["ID"][bi], data["turn_id"][bi],data["domain_focus"][bi],focus_dom_label_)                                                         
                            with open('save/dom-gate-error.txt', 'a+') as f:
                                 f.write(tosave_d + '\n')
                        if not torch.equal(GoldGates[bi], gating_pred):
                            tosave_s = "Dialog:{0},turn:{1}\nTrue:{2}\nPred:{3}\n".format(data["ID"][bi], data["turn_id"][bi],data["gating_label"][bi],gating_pred)                                                         
                            with open('save/slot-gate-error.txt', 'a+') as f:
                                 f.write(tosave_s + '\n')
                        if not torch.equal(GoldGates[bi], gating_pred_w_d): 
                            tosave_sr = "Dialog:{0},turn:{1}\nTrue:{2}\nPred:{3}\n".format(data["ID"][bi], data["turn_id"][bi], data["gating_label"][bi], gating_pred_w_d)                                                         
                            with open('save/slot-gate-REFINED-error.txt', 'a+') as f:
                                f.write(tosave_sr + '\n')
       
        
                         
        # decode
        decoded_out = self.decoder(previous_gold_y, gold_y, gold_y_idx, input_seq, input_len, \
                                   encoded_outputs, X_hdd, SLOT_hdd_all, PredictGates, \
                                   GoldGates, use_teacher_forcing, encoded_cfms) 
        
        all_point_outputs, words_point_out, updateGoldValue, updatePredValue, confirmGoldValue, confirmPredValue, bilinear = decoded_out
        
        if not self.decoder.training and classifiers_pred:
            for bi in range(batch_size):
                for si, sg in enumerate(GoldGates[bi, :]):
                    if sg == self.gating_dict["confirm"]:
                        SLOT_hdd = SLOT_hdd_all[bi, si, :]
                        SLOT_hdd_ = SLOT_hdd.expand_as(encoded_cfms)
                        p_cfm = bilinear(encoded_cfms.detach(), SLOT_hdd_).transpose(0, 1)
                        pred_class = torch.argmax(p_cfm, dim=1).item()
                        st = sub_gating_dict_verse[pred_class]
                        gold = gold_y[bi][si]
                        all_cfm_prediction['y_true'] += [sub_gating_dict.get(gold, len(sub_gating_dict))]
                        all_cfm_prediction['y_pred'] += [sub_gating_dict.get(st, len(sub_gating_dict))]

        gate_outs = (PredictGates, PredictInts)
        return gate_outs, decoded_out, predict_gates, predict_dom, all_slots_refine, classifiers_pred
        
    
    # run training operation of one batch in one Epoch
    def Train(self, data, reset=0):
        self.encoder.train()
        self.decoder.train()
        if reset: 
            self.reset() # put loss to 0
        self.optimizer.zero_grad()
        
        # args["teacher_forcing_ratio"] = 0.5
        use_teacher_forcing = random.random() < args["teacher_forcing_ratio"]

        gate_outs, decoded_out, predict_gates, predict_dom, all_slots_refine, _ = self.encode_and_decode(data, use_teacher_forcing)

        #encoded_outputs, X_hdd, SLOT_hdd_all, predict_gates, predict_dom = encoded_out

        all_point_outputs, words_point_out, updateGoldValue, updatePredValue, confirmGoldValue, confirmPredValue, _ = decoded_out

        # 计算Loss
        # loss_ptr
        assert len(updatePredValue) == len(updateGoldValue) and len(confirmPredValue) == len(confirmGoldValue)
        if len(updateGoldValue) > 0:
            #print("长度:",len(updateGoldValue)) # list; len = b * |J'| * m
            updateGoldValue_ = torch.tensor(updateGoldValue).contiguous() #torch.Size([len])
            updatePredValue_ = torch.stack(updatePredValue, dim=0).view(-1, self.vocab_size).contiguous()# torch.Size([len, |V|])
            loss_ptr = self.loss_func_ptr(updatePredValue_, updateGoldValue_)  
            #ptr_num_total = updateGoldValue_.ne(PAD_token_id).data.sum().item()
        else:
            loss_ptr = torch.tensor(0)
            #ptr_num_total = 0
        
        # loss_dom
        predict_dom = predict_dom.transpose(0, 1).contiguous().view(-1, predict_dom.size(-1))
        target_dom = data["domain_focus"].contiguous().view(-1)
        loss_dom = self.loss_func_dom(predict_dom, target_dom)
        
        #dom_num_total = predict_gates.size(1)
        
        # loss_gate
        predict_gates = predict_gates.transpose(0, 1).contiguous().view(-1, predict_gates.size(-1))
        target_gates = data["gating_label"].contiguous().view(-1)
        
        if args["training_gate_mode"] == 1: # training in subset of slots
            all_slots_refine = all_slots_refine.contiguous().view(-1)
            mask = (all_slots_refine != 0)
            #print(torch.equal(mask, all_slots_refine.bool()))
            predict_gates = predict_gates.masked_select(mask.unsqueeze(1))
            target_gates = target_gates.masked_select(mask)
        
        if len(target_gates) > 0:
            loss_gate = self.loss_func_opr(predict_gates, target_gates)
            #opr_num_total = target_gates.size(-1)
        else:
            loss_gate = torch.tensor(0)
            #opr_num_total = 0
        
        # loss_cfm
        if len(confirmGoldValue) > 0:
            true_confirm = torch.tensor(confirmGoldValue).contiguous() 
            predict_confirm = torch.stack(confirmPredValue, dim=0).contiguous().view(-1, len(sub_gating_dict)).contiguous()
            loss_cfm = self.loss_func_cfm(predict_confirm, true_confirm)
            #cfm_num_total = true_confirm.size(0)
        else:
            loss_cfm = torch.tensor(0)
            #cfm_num_total = 0
        

        # final loss
        loss = loss_ptr + loss_cfm + loss_gate + 2*loss_dom
        # (6) backward
        self.loss_grad = loss 

        self.loss += loss.data
        self.loss_ptr += loss_ptr.item()
        self.loss_cfm += loss_cfm.item()
        self.loss_gate += loss_gate.item()
        self.loss_dom += loss_dom.item()
        

    def evaluate(self, dev, matric_best, all_slots, early_stop=None):
        self.encoder.eval()
        self.decoder.eval()
        print("Start Evaluation...")
        print("training_gate_mode:", int(args['training_gate_mode'])) # default=0
        print("testing_gate_mode:", int(args['testing_gate_mode'])) # default=0

        all_prediction = {}
        all_dom_prediction = {'y_true': [], 'y_pred': []}
        all_slot_prediction = {'y_true': [], 'y_pred': []}
        all_slot_prediction_w = {'y_true': [], 'y_pred': []}
        all_cfm_prediction = {'y_true': [], 'y_pred': []}
        classifiers_pred = all_dom_prediction, all_slot_prediction, all_slot_prediction_w, all_cfm_prediction

        
        pbar = tqdm(enumerate(dev),total=len(dev))

        # outside loop: for each batch in DataLoader
        for j, data_dev in pbar: 
            # step 0:准备基本材料
            dev_batch_size, max_input_seq = data_dev['input_seq_idx'].size()
            # 1.encode and decode
            use_teacher_forcing = False 
            gate_outs, decoded_out, predict_gates, predict_dom, all_slots_refine, classifiers_pred = self.encode_and_decode(data_dev, use_teacher_forcing, classifiers_pred)
            PredictGates, PredictInts = gate_outs
            all_point_outputs, words_point_out, updateGoldValue, updatePredValue, confirmGoldValue, confirmPredValue, _ = decoded_out

            # inner loop: for each sample in the batch
            for bi in range(dev_batch_size):
                if data_dev["ID"][bi] not in all_prediction.keys():
                    all_prediction[data_dev["ID"][bi]] = {}
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {"turn_belief":data_dev["turn_belief"][bi]}
                predict_belief_bsz_ptr = []
                for si in range(len(all_slots)):
                    st = words_point_out[bi][si]
                    if st == "[NULL]": 
                        continue
                    else:
                        predict_belief_bsz_ptr.append(all_slots[si]+"-"+str(st))

                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr

                if set(data_dev["turn_belief"][bi]) != set(predict_belief_bsz_ptr) and args["genSample"]:
                    Trues, Preds = sorted(list((data_dev["turn_belief"][bi]))), sorted(predict_belief_bsz_ptr)
                    a = [x for x in Preds if x in Trues]
                    Trues_, Preds_ = [x for x in Trues if x not in a], [x for x in Preds if x not in a]
                    tolookup = "Dialog:{0},turn:{1}\nTrue:{2}\nPred:{3}\n".format(data_dev["ID"][bi], data_dev["turn_id"][bi],Trues_,Preds_)
                    with open('save/predict_errors.txt', 'a+') as f:
                         f.write(tolookup + '\n')

        print("The whole set is traversed. Results saved in dict [all_prediction]")
        if args["genSample"]:
            with open('save/all_cfm_prediction.pkl', 'wb') as f:
                pickle.dump(all_cfm_prediction, f, pickle.HIGHEST_PROTOCOL)
        # evaluate performance
        # classifier
        all_dom_prediction, all_slot_prediction, all_slot_prediction_w, all_cfm_prediction = classifiers_pred

        dom_cm, dom_joint_acc, dom_acc  = self.compute_gate(all_dom_prediction, turn_l=5, class_n=2)
        slot_cm, slot_joint_acc, slot_acc = self.compute_gate(all_slot_prediction, turn_l=30, class_n=3)
        slot_cmR, slot_joint_accR, slot_accR = self.compute_gate(all_slot_prediction_w, turn_l=30, class_n=3)
        cfm_cm, cfm_joint_acc, cfm_acc = self.compute_gate(all_cfm_prediction)
        # joint accuracy
        joint_acc_score_ptr, prf_score_ptr, turn_acc_score_ptr = self.evaluate_metrics(all_prediction, "pred_bs_ptr", all_slots)
        F1_score_ptr, r_score, p_score = prf_score_ptr
        print("Slot Opr Acc:{0}\n{1}".format(slot_joint_acc, slot_cm))
        print("Refined by Dom Opr:{0}\n{1}".format(slot_joint_accR, slot_cmR))
        print("Dom Opr Acc:{0}\n{1}".format(dom_joint_acc, dom_cm))
        print("Cfm Acc:{0}\n{1}".format(cfm_joint_acc, cfm_cm))
        print("Joint Acc:{:.4f}; Turn Acc:{:.4f}; Joint F1c:{:.4f}".format(joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr))
        print("Precision:{:.4f}; Recall :{:.4f}".format(p_score, r_score))
        joint_acc_score = joint_acc_score_ptr # (joint_acc_score_ptr + joint_acc_score_class)/2
        F1_score = F1_score_ptr

        self.encoder.train(True)
        self.decoder.train(True)

        if (early_stop == 'F1'):
            if (F1_score >= matric_best):
                self.save_model(directory = 'save')
                print("Model Saved...")  
            return F1_score
        else:
            if (joint_acc_score >= matric_best):
                self.save_model(directory = 'save')
                print("Model Saved...")
            return joint_acc_score
        

    def get_slots_refine(self, focus_dom_label_):
        slots_refine = [] 
        for di, dg in enumerate(focus_dom_label_):
            slots_refine += [int(dg)] * ALL_ds_nb[EXPERIMENT_DOMAINS[di]]
        assert len(slots_refine) == len(self.slots)
        return torch.tensor(slots_refine)

    def evaluate_metrics(self, all_prediction, from_which, slot_temp):
        total, turn_acc, joint_acc, F1_pred, p_pred, r_pred, Count = 0, 0, 0, 0, 0, 0, 0
        for d, v in all_prediction.items(): # d:dialog ID,eg:PMUL1635.json; v:dict
            #print("dialog:", d, "turns num:", len(v),v)
            #assert list(v.keys()) == list(range(len(v)))
            for t in range(len(v)):
                cv = v[t]
                if set(cv["turn_belief"]) == set(cv[from_which]):
                    joint_acc += 1
                total += 1

                # Compute prediction slot accuracy
                temp_acc = self.compute_acc(set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
                turn_acc += temp_acc

                # Compute prediction joint F1 score
                temp_f1, temp_r, temp_p, count = self.compute_prf(set(cv["turn_belief"]), set(cv[from_which]))
                F1_pred += temp_f1
                r_pred += temp_r
                p_pred += temp_p
                Count += count

        joint_acc_score = joint_acc / float(total) if total!=0 else 0
        turn_acc_score = turn_acc / float(total) if total!=0 else 0
        F1_score = F1_pred / float(Count) if Count!=0 else 0
        r_score = r_pred / float(Count) if Count!=0 else 0
        p_score = p_pred / float(Count) if Count!=0 else 0
        return joint_acc_score, (F1_score, r_score, p_score, ), turn_acc_score


    def compute_acc(self, gold, pred, slot_temp): # glod means groundtruth
        miss_gold = 0
        miss_slot = []
        for g in gold:
            if g not in pred: 
                miss_gold += 1
                miss_slot.append(g.rsplit("-", 1)[0]) 
        wrong_pred = 0
        for p in pred: 
            if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
                wrong_pred += 1
        ACC_TOTAL = len(slot_temp)
        # 1 - error_num / 30 = slot accuracy = ACC
        ACC = len(slot_temp) - miss_gold - wrong_pred
        ACC = ACC / float(ACC_TOTAL)
        return ACC

    def compute_prf(self, gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1 # FN=miss_gold
            for p in pred:
                if p not in gold:
                    FP += 1
            precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
            recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        else:
            if len(pred)==0:
                precision, recall, F1, count = 1, 1, 1, 1
            else:
                precision, recall, F1, count = 0, 0, 0, 1
        return F1, recall, precision, count


 
    def compute_gate(self, prediction, turn_l=1, class_n=None):
        y_true, y_pred = prediction["y_true"], prediction["y_pred"]
        labels = range(class_n) if class_n else None
        total, k = len(y_true), 0
        if total > 0:
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            #acc = sum([cm[i][i] for i in labels])/(sum([sum(cm[i]) for i in labels]))
            acc = accuracy_score(y_true, y_pred)
        else:
            cm = []
            acc = 0
        y_true = [y_true[i:i+turn_l] for i in range(0,len(y_true), turn_l)]
        y_pred = [y_pred[i:i+turn_l] for i in range(0,len(y_pred), turn_l)]
        for a,b in zip(y_true, y_pred):
            if a==b:
                k += 1
        joint_acc = k/total if total !=0 else 0
        return cm, joint_acc, acc
    
    '''
    def _refine(self, _predict_gates, slots_refine):
        #_predict_gates = predict_gates.transpose(0, 1)[bi] # [|s|, 3]
        gating_pred_w_d = torch.zeros(_predict_gates.size(0))
        for si in range(_predict_gates.size(-1)):
            k = slots_refine[si]
            if k==0:
                gating_pred_w_d[si] = 0
            else:
                #gating_pred_w_d[si] = torch.argmax(_predict_gates[si, k:], dim=-1)
                gating_pred_w_d[si] = torch.argmax(_predict_gates[si], dim=-1)
        return gating_pred_w_d.long()
    '''

    def demo(self, sample):
        """
        sample is a dict consists of :
        sample['previous_utterances'] = previous_utterances
        sample['current_utterances'] = current_utterances
        sample['previous_dict'] = previous_dict
        sample['input_seq'] = get_input_seq(previous_utterances, current_utterances, previous_dict)
        sample['previous_generate_y'] = get_generate_y(sample['previous_dict'])
        """
        print("start inference..")
        sample_data_set = LoadData([sample], word2id, 1)
        result = {}
        for i, data in enumerate(sample_data_set):
            batch_size, _ = data['input_seq_idx'].size()
            gate_outs, decoded_out, _, _, _, _ = self.encode_and_decode(data, False)            
            PredictGates, PredictInts = gate_outs
            words_point_out = decoded_out[1]
            for bi in range(batch_size):
                gating_pred = PredictGates[bi]
                interest_dom_label_ = PredictInts[bi]
                D = [d for di, d in enumerate(EXPERIMENT_DOMAINS) if interest_dom_label_[di] == 1]
                S = [s for si, s in enumerate(ALL_SLOTS) if gating_pred[si].item() != 0]
                V = [s + '-' + words_point_out[bi][si] for si, s in enumerate(ALL_SLOTS) if words_point_out[bi][si] != "[NULL]"]            
                result["domain"] = set(D)
                result["slot"] = set(S)
                result["value"] = set(V)
        return result

class EncoderBERT(nn.Module):
    def __init__(self, hidden_size, slots, gating_dict, vocab_size=None,):
        super(EncoderBERT, self).__init__()
        self.transformer = bmodel

        self.hidden_size = hidden_size
        self.slots = slots
        self.gating_dict = gating_dict
        self.nb_gate = len(gating_dict)
        self.nb_domain = len(EXPERIMENT_DOMAINS)
        # 1.predict_dom
        self.W_dom = nn.Linear(self.hidden_size*2, 2) # W_domain
        self.W_opr = nn.Linear(self.hidden_size, self.nb_gate)  # W_opration

        self.sigmoid = nn.Sigmoid()
        

        
    def encode_confirm_seq(self):
        # encode yes, NULL, do not care additionally
        encoded_cfms = torch.zeros(len(sub_gating_dict), self.hidden_size)
        for v_class, i in sub_gating_dict.items():
            sequence = tokenizer.encode(CLS_token + " " + v_class + " " + SEP_token)
            #print(sequence)
            sequence = torch.tensor(sequence).unsqueeze(0)
            H, v_X_hdd = self.transformer(sequence)
            #encoded_cfms[i, :] = v_X_hdd
            #print(H.size())
            encoded_cfms[i, :] = H[:, 0, :]
        return encoded_cfms
        
    # 3个工具函数    
    def _make_aux_tensors(self, input_seq, input_len, SEP_indices):
        # calculate position&tokentype by input_seq
        batch_size = input_seq.size(0)
        token_type_ids = torch.zeros(input_seq.size(), dtype=torch.long)
        for bi in range(batch_size):
            first_sep_pos = SEP_indices[bi, 0]
            token_type_ids[bi, first_sep_pos + 1: ] = 1
        attention_mask = input_seq > PAD_token_id #set PADs part as  0
        assert torch.equal(attention_mask.sum(dim = 1), torch.tensor(input_len)), "dislocation of [PAD]"
        return token_type_ids, attention_mask.long() # turn to torchLong type, or some bugs may jump out

    def get_token_position(self, input_seq, token_str, batch_size, start=None):
        _input_seq = input_seq.clone()
        if start == None:
            start = [0]*batch_size
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        pos_list = []
        for bi in range(batch_size):
            startPos = start[bi]
            _input_seq[bi, :startPos] = 0 
            try:
                pos = (_input_seq[bi, :] == token_id).nonzero().squeeze(1)
            except:
                pos = _input_seq[bi, :].size()[0]
            pos_list.append(pos)
        return torch.stack(pos_list, 0)


        
    def forward(self, input_seq, input_seq_len, input_max, SEP_indices):
        batch_size = input_seq.size(0)
        # 1. encoding
        if args['fix_embedding']:
            for p in self.encoder.pooler.parameters():
                p.requires_grad = False
        
        token_type_ids, attention_mask = self._make_aux_tensors(input_seq, input_seq_len, SEP_indices)
        
        last_hidden_state, pooler_output = self.transformer(input_seq, token_type_ids, attention_mask,position_ids=None, head_mask=None)
        encoded_outputs, X_hdd = last_hidden_state, pooler_output

        # 2. operation prediction
        predict_gates = torch.zeros(len(self.slots), batch_size, self.nb_gate) 
        
        start = [t[1].item() for t in SEP_indices]

        # step1: domain prediction
        # train W_dom
        Dom_hdd_all = torch.zeros(batch_size, self.nb_domain, self.hidden_size*2) # [4, 5, 768] 
        dom_pos = self.get_token_position(input_seq, DOM_token, batch_size, start)
        assert dom_pos.size() == torch.Size([batch_size, 5])
        dom_name_pos = dom_pos + 1
        #dom_pos = dom_pos - 1
        for bi in range(batch_size):
            H = encoded_outputs[bi, :, :] # [|X|, d]
            for di, dom in enumerate(EXPERIMENT_DOMAINS):
                Dom_hdd = H[dom_pos[bi, di].item(), :]
                Dom_name_hdd = H[dom_name_pos[bi, di].item(), :]
                #Dom_hdd_all[bi, di, :] = Dom_hdd + Dom_name_hdd
                Dom_hdd_all[bi, di, :] = torch.cat([Dom_hdd, Dom_name_hdd], 0)

        #predict_dom = torch.zeros(self.nb_domain, batch_size, 2) 
        #for di, dom in enumerate(EXPERIMENT_DOMAINS):
            #Dom_hdd_di = Dom_hdd_all[:, di, :]
            # add a sigmoid layer to enhance the 2-bit classfication。
            #predict_dom[di] = self.sigmoid(self.W_dom(Dom_hdd_di))
        predict_dom = self.sigmoid(self.W_dom(Dom_hdd_all)).permute(1,0,2)

        # step2: slot prediction
        SLOT_hdd_all = torch.zeros(batch_size, len(self.slots), self.hidden_size) # [4, 30, 768]
        slot_pos = self.get_token_position(input_seq, SLOT_token, batch_size, start)
        assert slot_pos.size() == torch.Size([batch_size, len(self.slots)])
        for bi in range(batch_size):
            H = encoded_outputs[bi, :, :] # [|X|, d]
            for si, slot in enumerate(self.slots):
                si_p = slot_pos[bi, si]
                SLOT_hdd = H[si_p, :] # d
                SLOT_hdd_all[bi, si, :] = SLOT_hdd
        
        # train W_opr
        for si, slot in enumerate(self.slots):
            # 模式1: slotGate训练时dom-focus不可见
            SLOT_hdd_si = SLOT_hdd_all[:, si, :]
            predict_gates[si] = self.W_opr(SLOT_hdd_si)

       
        return encoded_outputs, X_hdd, SLOT_hdd_all, predict_gates, predict_dom

class Generator(nn.Module):
    # all_point_outputs: [|s|, b, max_res_len, vocab_size]
    def __init__(self, vocab, vocab_size, hidden_size, dropout, slots, gating_dict, shared_emb):
        super(Generator, self).__init__()
        self.slots = slots
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.nb_gate = len(gating_dict)
        self.hidden_size = hidden_size
        self.gating_dict = gating_dict

        self.embedding = shared_emb # token embedding matrix shared with encoders
    
        self.dropout_layer = nn.Dropout(dropout)
        self.bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
        
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)        
        self.W_ratio = nn.Linear(3*hidden_size, 1) # W_1
        
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, previous_gold_y, gold_y, gold_y_idx, input_seq, input_len, encoded_outputs, X_hdd, SLOT_hdd_all, PredictGates, GoldGates, use_teacher_forcing, encoded_cfms):
        # input:
        # encoded_outputs: H_t size:[b, |X|, d]
        # X_hdd: h^{X}, initial hidden state, size:[b, d]
        # SLOT_hdd_all: |s| * h^{SLOT}, [b, |s|, d]
        # SLOT_hdd: h^{SLOT}, initial input, size:[b, d]
        # emb_mat: vocab matrix E
        max_res_len = 15 if not self.training else gold_y_idx.size(2)
        batch_size, max_input_seq = input_seq.size()

        all_point_outputs = torch.zeros(len(self.slots), batch_size, max_res_len, self.vocab_size) # [|s|, b, max_res_len, vocab_size]  = [30, 32, max_res_len, 30522]        
        words_point_out = []
        updateGoldValue, updatePredValue = [], []
        confirmGoldValue, confirmPredValue = [], []
        """
        if args["decoding_full_set"]: # default:0
            for si, slot in enumerate(self.slots):
                for bi in range(batch_size):
                    if self.training: # True
                        sg = data["gating_label"][bi, si]
                    else:
                        sg = PredictGates[bi, si]
                    SLOT_hdd = SLOT_hdd_all[bi, si, :]
                    if sg == self.gating_dict["confirm"]:
                        SLOT_hdd_ = SLOT_hdd.expand_as(encoded_cfms)
                        p_cfm = self.bilinear(encoded_cfms.detach(), SLOT_hdd_).transpose(0, 1)
                        pred_class = torch.argmax(p_cfm, dim=1).item()
                        st = sub_gating_dict_verse[pred_class]
                        gold = sub_gating_dict[gold_y[bi][si]]
                        confirmGoldValue.append(gold)
                        confirmPredValue.append(p_cfm)

                H = encoded_outputs
                hidden = X_hdd.expand(batch_size, 1, -1)
                decoder_input = self.dropout_layer(SLOT_hdd).expand(batch_size, self.hidden_size)
                encoded_lens = torch.tensor(input_len)
                story = input_seq
                sts_list = [[] for i in range(batch_size)]
                for wi in range(max_res_len):
                    dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)
                    context_vec, _, prob = self.attend(H, hidden.squeeze(0), encoded_lens)
                    p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                    p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                    vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec)) 
                    p_context_ptr = torch.zeros(p_vocab.size())
                    p_context_ptr.scatter_add_(1, story, prob) #p_context_ptr是论文里的 P_{ctx}
                    final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                            vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                    
                    pred_words = torch.argmax(final_p_vocab, dim=1) # size [1] 词表中id。TRADE里对隐的size是 ([32])
                    assert pred_words.size() == torch.Size([batch_size]), "Pred_word size Error."
                    #st_list.append(self.vocab[pred_word.item()]) 
                    #sts_list.append([self.vocab[pred_word.item()] for pred_word in pred_words])
                    all_point_outputs[si, :, wi, :] = final_p_vocab
                    for bi in range(batch_size):
                        pred_word = pred_words[bi]
                        st_list = sts_list[bi]
                        st_list.append(self.vocab[pred_word.item()])
                    if use_teacher_forcing:
                        decoder_input = self.embedding(gold_y_idx[:, si, wi]) # Chosen word is next input
                    else:
                        decoder_input = self.embedding(pred_words)
                words_out = []
                for bi in range(batch_size):  
                    st_list = sts_list[bi]
                    st_ = self.convert_tokens_to_string(st_list)
                    st = st_.split("[EOS]")[0].strip()
                    words_out.append(st) 
                words_point_out.append(words_out)
            words_point_out = np.asarray(words_point_out).transpose(1,0).tolist()
        """

        for bi in range(batch_size):
            if self.training:
                gates_opr = GoldGates[bi]
            else:
                gates_opr = PredictGates[bi] # gate torch.Size([|s|])=torch.Size([30]) 
            # 声明2个容器
            words_out = []
            H = encoded_outputs[bi, :, :].unsqueeze(0) 
            
            # H: (batch_size, |X|, hidden_size)=(1, |X|, d)
            for si, slot in enumerate(self.slots):
                sg = gates_opr[si] 
                SLOT_hdd = SLOT_hdd_all[bi, si, :]
                #gating_dict = {"carryover":0, "confirm":1, "update":2}
                if sg == self.gating_dict["carryover"]: # carry value from previous turn
                    st = previous_gold_y[bi][si]
                elif sg == self.gating_dict["confirm"]:
                    SLOT_hdd_ = SLOT_hdd.expand_as(encoded_cfms)
                    p_cfm = self.bilinear(encoded_cfms.detach(), SLOT_hdd_).transpose(0, 1)
                    pred_class = torch.argmax(p_cfm, dim=1).item()
                    st = sub_gating_dict_verse[pred_class]
                    gold = sub_gating_dict.get(gold_y[bi][si], len(sub_gating_dict))
                    if self.training:
                        assert gold in range(len(sub_gating_dict)), "CFM class label Error."
                        confirmGoldValue.append(gold)
                    else:
                        confirmGoldValue.append(0)
                    confirmPredValue.append(p_cfm)
                else:
                    decoder_input = self.dropout_layer(SLOT_hdd.expand(1, 1, -1)) 
                    decoder_input = decoder_input.squeeze(1)
                    hidden = X_hdd[bi, :].expand(1, 1, -1)
                    # copy
                    encoded_lens = torch.tensor(input_len)[bi].unsqueeze(0) # (1,1)
                    story = input_seq[bi].unsqueeze(0) # (1,|X|)
                    st_list = []
                    # same as TRADE, but b=1
                    # decoder_input:(seq_len, batch, input_size)=(1,1,d)
                    # hidden: (n_layers * num_directions, batch, hidden_size)=(1,1,d)
                    #print("max_res_len:", max_res_len)
                    for wi in range(max_res_len):
                        # hidden    : g_k, next step's input hidden
                        # dec_state 
                        # since seq_len=1, n_layer=1, direction_n=1, 'dec_state' is euqal with 'hidden'。
                        dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)

                        # (1)context_vec: the c in paper, size=(b, d);  prob: the P_ctx in paper, size=(b, |X|)
                        context_vec, _, prob = self.attend(H, hidden.squeeze(0), encoded_lens)
                        # (2)get P_{vcb}
                        # shared_emb.weight: [vocab_size, embedding_size] = [30522, 768]
                        p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                        # (3)calculate alpha/p_gen/generate probability
                        # size=(b, d) * 3
                        p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                        # alpha = vocab_pointer_switches = generate probability , (torch.size([1, 1]))
                        # alpha is a scalar
                        vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec)) 

                        p_context_ptr = torch.zeros(p_vocab.size())
                        # scatter_add_: copy
                        # from prob to p_context_ptr: (1,|X|)-->(1,|V|)
                        p_context_ptr.scatter_add_(1, story, prob) #p_context_ptr is the P_{ctx} in paper
                        # (4)get P_{val}
                        final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                                vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                        
                        pred_word = torch.argmax(final_p_vocab, dim=1) 
                        assert pred_word.size() == torch.Size([1]), "Pred_word size Error."
                        # (4) put results into caontainer
                        #print(self.vocab[pred_word.item()]) # str:token
                        st_list.append(self.vocab[pred_word.item()]) 
                        #print(wi, pred_word, self.vocab[pred_word.item()])

                        if self.training: 
                            updateGoldValue.append(gold_y_idx[bi, si, wi].item())
                        else:
                            updateGoldValue.append(0)
                        updatePredValue.append(final_p_vocab)
                        # (5) prepare the next decoding step
                        if use_teacher_forcing:
                            decoder_input = self.embedding(gold_y_idx[bi, si, wi].unsqueeze(0)) # Chosen word is next input

                        else:
                            decoder_input = self.embedding(pred_word)   
                            if USE_CUDA: decoder_input = decoder_input.cuda()
                    #st_ = tokenizer.convert_tokens_to_string(st_list)
                    st_ = self.convert_tokens_to_string(st_list)
                    st = st_.split("[EOS]")[0].strip()

                words_out.append(st)

            words_point_out.append(words_out)

        return all_point_outputs, words_point_out, updateGoldValue, updatePredValue, confirmGoldValue, confirmPredValue, self.bilinear
 
    def attend(self, seq, cond, lens):
        # get P_{ctx} and c
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2) # size:(32, 2xx)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf # set PAD part as -np.inf
        scores = F.softmax(scores_, dim = 1) # size:(b, |X|)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores
    
    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        scores = F.softmax(scores_, dim = 1)
        return scores

    """
    def get_att_dis(self, target, multis):
        attention_distribution = []
        for i in range(multis.size(0)):
            attention_score = torch.cosine_similarity(target, multis[i].view(1, -1))
            attention_distribution.append(attention_score)
        attention_distribution = torch.Tensor(attention_distribution)
        return attention_distribution / torch.sum(attention_distribution, 0) 
    """
    
    # you could choose to do post-processing for a few acceptable error cases
    def convert_tokens_to_string(self, tokens, post_processing=True):
        out_string = ' '.join(tokens)
        out_string = out_string.replace(' ##', '').replace(' : ', ':').strip()
        if post_processing:
            sre = re.search(r"(\d{1,2}:\d{1,2})", out_string)
            if sre:
                return sre.group().zfill(5).strip()
            out_string = out_string.replace('center', 'centre')
            return out_string
        else:
            return out_string
        




