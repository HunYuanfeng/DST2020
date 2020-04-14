## Gate-Enhanced Multi-Domain Dialog State Tracking for Task-Oriented Dialogue Systems

**\*\*\*\*\* March 30th, 2020 \*\*\*\*\***

It is the original PyTorch implementation to this paper:
**Gate-Enhanced Multi-Domain Dialog State Tracking for Task-Oriented Dialogue Systems**

The code is written and tested with PyTorch == 1.1.0. 

Our experiments mainly conduct on [MultiWOZ 2.1](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip) corpus, and the 2.0 verison is also supported.

## Requirements

- python 3.6
- pytorch >= 1.1.0
- Install python packages:
  
  - `pip install transformers`
  
    (The pretrained BERT Models need to be downloaded [here](https://github.com/huggingface/transformers) in advance.)

## Download Data

```
python3 create_data.py 
```
This Python script inherits from the same-name script in [TRADE](https://github.com/jasonwu0731/trade-dst) with minor modifications.

***************************************************************


## Preprocessing
```
python3 make_dataset.py
```

***************************************************************

## Training
```
python3 run.py
```

- -dataset: which corpus to train
- -path: model saved path
- -train_m: training gate mode

****************************************************************

## Evaluation
```
python3 run_test.py -path=${save_path}
```

- -dataset: which corpus to train
- -path: model saved path
- -test_m: testing gate mode

## Contact Information

Contact: Changhong Yu (`CharpYu@bupt.edu.cn`)

