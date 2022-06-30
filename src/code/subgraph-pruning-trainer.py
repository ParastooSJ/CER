import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger("Qblink-20-bert-regressor")

FP16 = False
# BATCH_SIZE = 32
BATCH_SIZE = 16
SEED = 42


WARMUP_PROPORTION = 0.1
PYTORCH_PRETRAINED_BERT_CACHE = "./cache"
LOSS_SCALE = 0.
MAX_SEQ_LENGTH = 100


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


import random
import numpy as np

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if n_gpu > 0:
    torch.cuda.manual_seed_all(SEED)

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

    def __init__(self, question,sentence,c_et_glossory,score=None):
        self.question=question
        self.sentence=sentence
        self.c_et_glossory=c_et_glossory
        self.score=score
    
 

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids,score):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.score = score

"""Data Processing Class and Function:"""

df_train = pd.read_json('../data/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic_random_train_scored.json', lines=True)

df_test = pd.read_json('../data/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic_random_train_scored.json', lines=True)

df_dev = pd.read_json('../data/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic_random_train_scored.json', lines=True)



class DataProcessor:
    

    question_train=[]
    sentence_train=[]
    c_et_glossory_train=[]
    score_train=[]

    question_test=[]
    sentence_test=[]
    c_et_glossory_test=[]
    score_test=[]


    question_dev=[]
    sentence_dev=[]
    c_et_glossory_dev=[]
    score_dev=[]


    def __init__(self):
        
        for _, value in df_train['question'].iteritems():
            self.question_train.append(value[:1000].strip())
            
        for _, value in df_train['triple'].iteritems():
            self.sentence_train.append(value['sentence'][:1000].strip())
            self.c_et_glossory_train.append(value['c_et'][:1000].strip())
            self.score_train.append(value['score'])
            
            
        for _, value in df_test['question'].iteritems():
            self.question_test.append(value[:1000].strip())
        for _, value in df_test['triple'].iteritems():
            self.sentence_test.append(value['sentence'][:1000].strip())
            self.c_et_glossory_test.append(value['c_et'][:1000].strip())
            self.score_test.append(value['score'])
            
            
        for _, value in df_dev['question'].iteritems():
            self.question_dev.append(value[:1000].strip())
        for _, value in df_dev['triple'].iteritems():
            self.sentence_dev.append(value['sentence'][:1000].strip())
            self.c_et_glossory_dev.append(value['c_et'][:1000].strip())
            self.score_dev.append(value['score'])
        
        self.score_train=(self.score_train-np.min(self.score_train))/(np.max(self.score_train)-np.min(self.score_train))
        self.score_dev=(self.score_dev-np.min(self.score_dev))/(np.max(self.score_dev)-np.min(self.score_dev))
        self.score_test=(self.score_test-np.min(self.score_test))/(np.max(self.score_test)-np.min(self.score_test))
        
    
    def get_dev_examples(self):
        return self._create_examples(self.question_dev,self.sentence_dev,self.c_et_glossory_dev,self.score_dev)
        
        
    def get_train_examples(self):
        return self._create_examples(self.question_train,self.sentence_train,self.c_et_glossory_train,self.score_train)

   

    def get_test_examples(self):
        return self._create_examples(self.question_test,self.sentence_test,self.c_et_glossory_test,self.score_test)

    def _create_examples(self,question,sentence,c_et_glossory,score):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, (question,sentence,c_et_glossory,score)) in enumerate(zip(question,sentence,c_et_glossory,score)):
            examples.append(
                InputExample(question=question,sentence=sentence,c_et_glossory=c_et_glossory,score=score))
        return examples

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
                              segment_ids=segment_ids,
                              score=example.score))
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

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True, 
    cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)

train_examples = DataProcessor().get_train_examples()

train_features = convert_examples_to_features(
    train_examples, MAX_SEQ_LENGTH, tokenizer)
del train_examples
gc.collect()

"""### Model definition"""

# Prepare model
model = BertForSequenceRegression.from_pretrained(
    'bert-base-uncased',
    cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
if FP16:
    model.half()
model.to(device)

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

"""### The Training Loop"""

def train(model: nn.Module, num_epochs: int, learning_rate: float):
    num_train_optimization_steps = len(train_dataloader) * num_epochs 
    optimizer = get_optimizer(num_train_optimization_steps, learning_rate)
    assert all([x["lr"] == learning_rate for x in optimizer.param_groups])
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", BATCH_SIZE)
    logger.info("  Num steps = %d", num_train_optimization_steps)    
    model.train()
    mb = master_bar(range(num_epochs))
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0    
    for _ in mb:
        for step, batch in enumerate(progress_bar(train_dataloader, parent=mb)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, score = batch

            loss = model(input_ids, segment_ids, input_mask, score)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.

            if FP16:
                optimizer.backward(loss)
            else:
                loss.backward()

            if tr_loss == 0:
                tr_loss = loss.item()
            else:
                tr_loss = tr_loss * 0.9 + loss.item() * 0.1
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if FP16:
                # modify learning rate with special warm up BERT uses
                # if args.fp16 is False, BertAdam is used that handles this automatically
                lr_this_step = (
                     LR * warmup_linear(global_step/num_train_optimization_steps, WARMUP_PROPORTION))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            mb.child.comment = f'loss: {tr_loss:.4f} lr: {optimizer.get_lr()[0]:.2E}'
    logger.info("  train loss = %.4f", tr_loss) 
    return tr_loss

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_scores = torch.tensor([f.score for f in train_features], dtype=torch.float)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_scores)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# Train only the "pooler" and the final linear layer
set_trainable(model, True)
set_trainable(model.bert.embeddings, False)
set_trainable(model.bert.encoder, False)
count_model_parameters(model)
train(model, num_epochs = 2, learning_rate = 5e-4)

# Save a trained model
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = "../model/"+sys.argv[1]+"/"+sys.argv[1]+"_fold_"+sys.argv[2]+"/basic_bert_regressor1.pth"
#torch.save(model_to_save.state_dict(), output_model_file)


gc.collect()

# Train the last two layer, too
set_trainable(model.bert.encoder.layer[11], True)
set_trainable(model.bert.encoder.layer[10], True)
count_model_parameters(model)
train(model, num_epochs = 1, learning_rate = 5e-5)

# Save a trained model
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = "../model/"+sys.argv[1]+"/"+sys.argv[1]+"_fold_"+sys.argv[2]+"/basic_bert_regressor2.pth"
torch.save(model_to_save.state_dict(), output_model_file)

model.load_state_dict(torch.load(output_model_file))
gc.collect()

# Train all layers
set_trainable(model, True)
count_model_parameters(model)
#train(model, num_epochs = 1, learning_rate = 1e-5)

# Save a trained model
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = "../model/"+sys.argv[1]+"/"+sys.argv[1]+"_fold_"+sys.argv[2]+"/basic_bert_regressor3.pth"
torch.save(model_to_save.state_dict(), output_model_file)



# Load a trained model that you have fine-tuned
# model_state_dict = torch.load(output_model_file)
# model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=num_labels)
# model.to(device)

# del train_features
gc.collect()

model=BertForSequenceRegression.from_pretrained('bert-base-uncased',cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
model.load_state_dict(torch.load("../model/"+sys.argv[1]+"/"+sys.argv[1]+"_fold_"+sys.argv[2]+"/basic_bert_regressor3.pth"))

model.to(device)

"""## Evauluation"""

eval_examples = DataProcessor().get_dev_examples()

eval_features = convert_examples_to_features(eval_examples, MAX_SEQ_LENGTH, tokenizer)

logger.info("***** Running evaluation *****")
logger.info("  Num examples = %d", len(eval_examples))
logger.info("  Batch size = %d", BATCH_SIZE * 5)
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_scores = torch.tensor([f.score for f in eval_features], dtype=torch.float)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_scores)
# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=BATCH_SIZE * 5)

model.eval()
eval_loss, eval_accuracy = 0, 0

# all_scores,all_outputs=[],[]
counter=0

nb_eval_steps, nb_eval_examples,mse, r_square,mape = 0, 0,0,0,0

mb = progress_bar(eval_dataloader)
for input_ids, input_mask, segment_ids, scores in mb:
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    scores = scores.to(device)

    with torch.no_grad():
        tmp_eval_loss = model(input_ids, segment_ids, input_mask,scores)
        outputs = model(input_ids, segment_ids, input_mask)

    outputs = outputs.detach().cpu().numpy()
    scores = scores.to('cpu').numpy()
    
    if counter==0:
        all_scores=torch.from_numpy(scores)
        all_outputs=torch.from_numpy(outputs)
        counter+=1
    else:
        all_scores=torch.cat([all_scores, torch.from_numpy(scores)], dim=0)
        all_outputs=torch.cat([all_outputs, torch.from_numpy(outputs)], dim=0)
    
    
    
    eval_loss += tmp_eval_loss.mean().item()


    nb_eval_examples += input_ids.size(0)
    nb_eval_steps += 1
    mb.comment = f'{eval_loss / nb_eval_steps:.4f}'

print("loss:")
print(eval_loss / nb_eval_steps)


#mse = mean_squared_error(all_scores, all_outputs)
#print("Mean Squared Error :",mse)

#mape=mean_absolute_percentage_error(all_scores, all_outputs)
#print("Mean Absolute Percentage Error:",mape)

#r_square= r2_score(all_scores, all_outputs)
#print("R^2 :",r_square)




















