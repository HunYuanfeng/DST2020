import os
import logging 
import argparse
from tqdm import tqdm
import torch

if torch.cuda.is_available():
    USE_CUDA = True
else:
    USE_CUDA = False

parser = argparse.ArgumentParser(description='YCH Multi-Woz 2.1 DST')

# Training Setting
parser.add_argument('-ds','--dataset', help='multiwoz2.0 or multiwoz2.1', required=False, default="multiwoz2.1")
parser.add_argument('-path','--path', help='path of the modelfile to load', required=False,default=False)
parser.add_argument('-sample','--sample', help='Number of Samples', required=False,default=None)
parser.add_argument('-patience','--patience', help='', required=False, default=6, type=int)
parser.add_argument('-es','--earlyStop', help='Early Stop Criteria, BLEU or ENTF1', required=False, default='BLEU')
parser.add_argument('-imbsamp','--imbalance_sampler', help='', required=False, default=0, type=int)
parser.add_argument('-um','--unk_mask', help='mask out input token to UNK', type=int, required=False, default=1)
parser.add_argument('-bsz','--batch_size', help='Training Batch_size', required=False, type=int, default=16)
parser.add_argument('-epoch','--epoch_num', help='Training epochs number', required=False, type=int, default=3)

# Testing Setting
parser.add_argument('-viz','--vizualization', help='vizualization', type=int, required=False, default=0)
parser.add_argument('-gs','--genSample', help='Generate Sample', type=int, required=False, default=0)
parser.add_argument('-evalp','--evalp', help='evaluation period', required=False, default=1)
parser.add_argument('-eb','--eval_batch', help='Evaluation Batch_size', required=False, type=int, default=4)
parser.add_argument('-ws','--use_weighted_sampler', help='sample weighted in dataloader', required=False, type=bool, default=False)
parser.add_argument('-dfs',"--decoding_full_set", help='run SpanDecoding on all slots', required=False, type=int, default=0) 

# Model architecture
'''
traning_gate_mode: 
训练时, slot gate有2种mode(训练模式)
1.照常训练, 即SOM-DST的方法； training_gate_mode = 0
2.利用dom的oracle来辅助slot opr的训练； training_gate_mode = 1(此情况下, testing模式不可以是0, 默认是2)
testing_gate_mode: 
测试时, slot gate有4种mode(测试模式)
1.照常预测, 即SOM-DST的方法； testing_gate_mode = 0
2.利用slot的oracle来预测； testing_gate_mode = 1
3.利用dom的predict来辅助slot opr的预测； testing_gate_mode = 2
4.利用dom的oracle来辅助slot opr的预测； testing_gate_mode = 3
----------------------
整合:
gate_mode = 0:使用子集机制训练, 即原训练模式1，验证模式2
训练：利用dom的oracle来辅助slot opr的训练
验证：利用dom的predict来辅助slot opr的预测
gate_mode = 1:不使用子集训练，即原训练模式0，验证模式0
训练：照常训练, 即SOM-DST的方法
验证：照常预测, 即SOM-DST的方法
gate_mode = 2:
'''

parser.add_argument('-femb','--fix_embedding', help='', required=False, default=0, type=int)
parser.add_argument('-train_m','--training_gate_mode', help='2 kinds of mode for training slot-gate', required=False, default=0, type=int)
parser.add_argument('-test_m','--testing_gate_mode', help='4 kinds of mode for testing slot-gate', required=False, default=0, type=int)
parser.add_argument('-runtest','--run_test_testing', help='run testing in test set', required=False, default=1, type=int)
parser.add_argument('-rundev','--run_dev_testing', help='', required=False, default=0, type=int)


# Model Hyper-Parameters
parser.add_argument('-hdd','--hidden_size', help='Hidden size', required=False, type=int, default=768)
parser.add_argument('-lr','--learning_rate', help='Learning Rate', required=False, type=float)
parser.add_argument('-dr','--dropout', help='Drop Out', required=False, type=float, default=0.1)
parser.add_argument('-lm','--limit', help='Word Limit', required=False,default=-10000)
parser.add_argument('-clip','--clip', help='gradient clipping', required=False, default=10, type=int) 
parser.add_argument('-tfr','--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False, default=0.5)
parser.add_argument('-warm','--warmup_prop', help='warmup proporation', required=False, type=float, default=0.1)


args = vars(parser.parse_args())
#print(str(args))
if args['training_gate_mode']==1 and args['testing_gate_mode']==0:
    args['testing_gate_mode'] = 2

assert not args['testing_gate_mode'] < args['training_gate_mode']
