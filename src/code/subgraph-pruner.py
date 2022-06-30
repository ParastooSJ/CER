# -*- coding: utf-8 -*-
"""scoring_Qblink_with_Bert_learned_on _20_sample.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10QmGeyS2z9wGb5H-ccn6Hx20sWchK6Xy

Adapted from [this example script](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py)

Logger settings and Constants:
"""

import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger("Qblink-50-bert-regressor")

FP16 = False
BATCH_SIZE = 16
SEED = 42


WARMUP_PROPORTION = 0.1
LOSS_SCALE = 0. # Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.
MAX_SEQ_LENGTH = 100

"""## Imports"""

import gc

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule, SCHEDULES
from fastprogress import master_bar, progress_bar
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_percentage_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(
    device, n_gpu, FP16))

"""Set random seeds:"""

import random
import numpy as np

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if n_gpu > 0:
    torch.cuda.manual_seed_all(SEED)

"""## Definitions

Regression model:
"""

class BertForSequenceRegression(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceRegression, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)
        self.loss_fct = torch.nn.MSELoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, targets=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        outputs = self.regressor(pooled_output).clamp(-1, 1)
        if targets is not None:
            loss = self.loss_fct(outputs.view(-1), targets.view(-1))
            return loss
        else:
            return outputs

"""Data Classes:"""

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, question,sentence,c_et_glossory):
        self.question=question
        self.sentence=sentence
        self.c_et_glossory=c_et_glossory
#         self.score=score
    
 

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
#         self.score = score

class DataProcessor:
    """Processor for the data set."""
    def __init__(self):
        pass
    
    def get_dev_examples(self,df_dev):
        
        question_dev=[]
        sentence_dev=[]
        c_et_glossory_dev=[]

        for  value in df_dev['triples']:
            sentence_dev.append(value['sentence'][:1000].strip())
            c_et_glossory_dev.append(value['c_et'][:1000].strip())
            question_dev.append(df_dev['question'][:1000].strip())
        
        return self._create_examples(question_dev,sentence_dev,c_et_glossory_dev)
        
        
    def get_train_examples(self,df_train):
        
        question_train=[]
        sentence_train=[]
        c_et_glossory_train=[]
        score_train=[]
            
        for value in df_train['triples']:
            sentence_train.append(value['sentence'][:1000].strip())
            c_et_glossory_train.append(value['c_et'][:1000].strip())
            question_train.append(df_train['question'][:1000].strip())
        
        return self._create_examples(question_train,sentence_train,c_et_glossory_train)

   

    def get_test_examples(self,df_test):
        
        question_test=[]
        sentence_test=[]
        c_et_glossory_test=[]

        for value in df_test['triples']:
            sentence_test.append(value['sentence'][:1000].strip())
            c_et_glossory_test.append(value['c_et'][:1000].strip())
            question_test.append(df_test['question'][:1000].strip())
        
        return self._create_examples(question_test,sentence_test,c_et_glossory_test)

    def _create_examples(self,question,sentence,c_et_glossory):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, (question,sentence,c_et_glossory)) in enumerate(zip(question,sentence,c_et_glossory)):
            examples.append(
                InputExample(question=question,sentence=sentence,c_et_glossory=c_et_glossory))
        return examples

"""Data Processing Class and Function:"""

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    
    features = []
    for (ex_index, example) in enumerate(examples):
        question_tokens = tokenizer.tokenize(example.question)
        
        if len(question_tokens) > max_seq_length - 1:
            question_tokens = question_tokens[:(max_seq_length - 1)]

    
        question_tokens = ["[CLS]"] + question_tokens
        question_segment_ids = [0] * len(question_tokens)

        question_input_ids = tokenizer.convert_tokens_to_ids(question_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        question_input_mask = [1] * len(question_input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(question_input_ids))
        question_input_ids += padding
        question_input_mask += padding
        question_segment_ids += padding


        #--------------------------c_et_glossory----------------------------

        c_et_glossory_tokens = tokenizer.tokenize(example.c_et_glossory)
        
        if len(c_et_glossory_tokens) > max_seq_length - 2:
            c_et_glossory_tokens = c_et_glossory_tokens[:(max_seq_length - 2)]

        c_et_glossory_tokens = ["[SEP]"] + c_et_glossory_tokens + ["[SEP]"]
        c_et_glossory_segment_ids = [1] * len(c_et_glossory_tokens)

        c_et_glossory_input_ids = tokenizer.convert_tokens_to_ids(c_et_glossory_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        c_et_glossory_input_mask = [1] * len(c_et_glossory_input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(c_et_glossory_input_ids))
        c_et_glossory_input_ids += padding
        c_et_glossory_input_mask += padding
        c_et_glossory_segment_ids += padding

        #--------------------------all to gether----------------------------
        input_ids=question_input_ids+c_et_glossory_input_ids
        input_mask=question_input_mask+c_et_glossory_input_mask
        segment_ids=question_segment_ids+c_et_glossory_segment_ids

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features

class FreezableBertAdam(BertAdam):
    def get_lr(self):
        lr = []
        for group in self.param_groups:

            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    continue
                lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

"""Utility functions:"""

def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))

def count_model_parameters(model):
    logger.info(
        "# of paramters: {:,d}".format(
            sum(p.numel() for p in model.parameters())))
    logger.info(
        "# of trainable paramters: {:,d}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)))

"""## Training

### Preprocessing
"""
PYTORCH_PRETRAINED_BERT_CACHE = './cache'
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True, 
    cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)

gc.collect()
torch.cuda.empty_cache()

"""### Model definition"""

model=BertForSequenceRegression.from_pretrained('bert-base-uncased',cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
model.load_state_dict(torch.load("../model/"+sys.argv[1]+"/"+sys.argv[1]+"_fold_"+sys.argv[2]+"/basic_bert_regressor3.pth"))
model.to(device)

gc.collect()

# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

def get_optimizer(num_train_optimization_steps: int, learning_rate: float):
    grouped_parameters = [
       x for x in optimizer_grouped_parameters if any([p.requires_grad for p in x["params"]])
    ]
    for group in grouped_parameters:
        group['lr'] = learning_rate
    if FP16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex "
                              "to use distributed and fp16 training.")

        optimizer = FusedAdam(grouped_parameters,
                              lr=learning_rate, bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=LOSS_SCALE)

    else:
        optimizer = FreezableBertAdam(grouped_parameters,
                             lr=learning_rate, warmup=WARMUP_PROPORTION,
                             t_total=num_train_optimization_steps)
    return optimizer

"""## Evauluation"""

def get_score(eval_features,batch_size):
    
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    model.eval()
    counter=0
    
    mb = progress_bar(eval_dataloader)
    for input_ids, input_mask, segment_ids in mb:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, segment_ids, input_mask)

        outputs = outputs.detach().cpu().numpy()

        if counter==0:
            all_outputs=torch.from_numpy(outputs)
            counter+=1
        else:
            all_outputs=torch.cat([all_outputs, torch.from_numpy(outputs)], dim=0)
            
    return all_outputs

import json 
import ast
from sklearn.metrics import average_precision_score
labels =[]
batch_size=BATCH_SIZE * 25
#test_file = open("./data/qblink-test-all-scored.json",'w', encoding='utf-8')
#filename='./data/qblink_test_with_glossery.json'
test_file = open("../data/"+sys.argv[1]+"/"+sys.argv[1]+"_fold_"+sys.argv[2]+"/basic_test_scored.json",'w', encoding='utf-8')
filename='../data/'+sys.argv[1]+"/"+sys.argv[1]+"_fold_"+sys.argv[2]+"/basic_test_with_wiki.json"
counter=0
ap =0 
with open(filename, 'r') as f:
    for line in f:
        
        print("test: question "+str(counter)+" is processing")
        data = []
        data=json.loads(line)

        counter+=1

        test_examples = DataProcessor().get_test_examples(data)
        test_features = convert_examples_to_features(test_examples, MAX_SEQ_LENGTH, tokenizer)

        
        scores=get_score(test_features,batch_size)
        
        labels =[]
        for triple,score in zip(data["triples"],scores):
            triple["score"]=score.item()
            labels.append(1 if triple["rank"]>0 else 0)
        labels = np.array(labels)
        ap = ap + average_precision_score(labels, scores)
        test_file.write(json.dumps(data)+"\n")
print(float(ap)/counter)
print('--------------------------------------------------')
test_file.close()

import json 
import ast
ap = 0 
batch_size=BATCH_SIZE * 20
#train_file = open("./data/qblink-train-all-scored.json",'w', encoding='utf-8')
#filename='./data/qblink_train_with_glossery.json'
train_file = open("../data/unique/data/"+sys.argv[1]+"/"+sys.argv[1]+"_fold_"+sys.argv[2]+"/basic_train_scored.json",'w', encoding='utf-8')
filename='../data/unique/data/'+sys.argv[1]+"/"+sys.argv[1]+"_fold_"+sys.argv[2]+"/basic_train_with_wiki.json"
counter=0
with open(filename, 'r') as f:
    for line in f:
        print("train: question "+str(counter)+" is processing")
        
        data = []
        data=json.loads(line)

        counter+=1
        print(str(counter))
        train_examples = DataProcessor().get_train_examples(data)
        train_features = convert_examples_to_features(train_examples, MAX_SEQ_LENGTH, tokenizer)

        scores=get_score(train_features,batch_size)
        labels =[]
        for triple,score in zip(data["triples"],scores):
            triple["score"]=score.item()
            labels.append(1 if triple["rank"]>0 else 0)
        labels= np.array(labels)
        ap = ap + average_precision_score(labels, scores)
        train_file.write(json.dumps(data)+"\n")
print(float(ap)/counter)
train_file.close()

