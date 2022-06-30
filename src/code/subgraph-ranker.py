import logging
import gc

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule, SCHEDULES
from fastprogress import master_bar, progress_bar
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_percentage_error
import random
import numpy as np
import json
import pandas as pd
import numpy as np
import gc
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.modeling import BertPreTrainedModel ,BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule, SCHEDULES
from fastprogress import master_bar, progress_bar
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_percentage_error
import random
from sentence_transformers import SentenceTransformer, util
from transformers import EncoderDecoderModel, BertTokenizer,BertModel,AdamW,get_scheduler
import torch
from sklearn.utils import class_weight
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score
from sklearn.metrics import ndcg_score
import sys
FP16 = False
BATCH_SIZE = 2
SEED = 42
WARMUP_PROPORTION = 0.1
PYTORCH_PRETRAINED_BERT_CACHE = "../cache"
LOSS_SCALE = 0.
MAX_SEQ_LENGTH = 100

logger = logging.getLogger("Qblink-20-bert-regressor")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(
    device, n_gpu, FP16))
    
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if n_gpu > 0:
    torch.cuda.manual_seed_all(SEED)


#model
class BertForSequenceRegression(BertPreTrainedModel):
    def __init__(self,config):
        super(BertForSequenceRegression, self).__init__(config)
        self.bert_q = BertModel.from_pretrained('bert-base-uncased')
        self.bert_a = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.cos = torch.nn.CosineSimilarity(dim=1)

        self.linear = nn.Linear(2, 1)
        self.sig = nn.Sigmoid()
       
    def forward(self, input_ids, attention_mask,output_ids,output_attention_mask,label):
      q_output = self.bert_q(input_ids,attention_mask).pooler_output
      a_output = self.bert_a(output_ids,output_attention_mask).pooler_output
      
      join_output = self.cos(q_output, a_output)
      join_output = join_output.unsqueeze(-1)
      label = label.unsqueeze(-1)
      concat = torch.cat((join_output,label),dim=1)
      final = self.linear(concat)
      
      return final

class InputExample(object):
    def __init__(self,question,answer_pos,answer_neg,label_neg,label_pos):
        self.question=question
        self.answer_pos=answer_pos
        self.answer_neg = answer_neg
        self.label_neg = label_neg
        self.label_pos = label_pos

class InputFeatures(object):

    def __init__(self, input_ids, input_mask,output_pos_ids,output_pos_mask,output_neg_ids,output_neg_mask,label_neg,label_pos):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.output_pos_ids = output_pos_ids
        self.output_pos_mask = output_pos_mask
        self.output_neg_ids = output_neg_ids
        self.output_neg_mask = output_neg_mask
        self.label_neg = label_neg
        self.label_pos = label_pos

class DataProcessor:

    question_train=[]
    answer_pos_train=[]
    answer_neg_train=[]
    label_pos_train=[]
    label_neg_train=[]

    def __init__(self):
        x=2
    
    def get_train_examples(self):
        for id, value in df_train['triples'].iteritems():
          pos_answer= []
          pos_answer2=[]
          pos_rank1 =[]
          pos_rank2=[]
          

          for triple in value:
            if triple['rank']>=2:
              pos_answer2.append('centity '+triple['c_et']+' sentence '+triple['sentence']+' qentity '+triple['q_et'])
              pos_rank2.append(triple)
            elif triple['rank']>=1:
                pos_answer.append('centity '+triple['c_et']+' sentence '+triple['sentence']+' qentity '+triple['q_et'])
                pos_rank1.append(triple)
          if len(pos_answer)>0 or len(pos_answer2)>0:
            i = 0
            for pos in pos_answer:
                for triple in value:
                  if triple['ans']==False:
                    self.question_train.append(df_train['question'][id])
                    self.answer_neg_train.append('centity '+triple['c_et']+' sentence '+triple['sentence']+' qentity '+triple['q_et'])
                    self.answer_pos_train.append(pos)
                    self.label_neg_train.append(triple['score'])
                    self.label_pos_train.append(pos_rank1[i]['score'])
                i += 1
            i=0
            for pos in pos_answer2:
                for triple in value:
                    if triple['ans']==False:
                        self.question_train.append(df_train['question'][id])
                       # self.answer_neg_train.append(triple['c_et'])
                        self.answer_neg_train.append('centity '+triple['c_et']+' sentence '+triple['sentence']+' qentity '+triple['q_et'])
                        self.answer_pos_train.append(pos)
                        self.label_neg_train.append(triple['score'])
                        self.label_pos_train.append(pos_rank2[i]['score'])
                i +=1

        return self._create_examples(self.question_train,self.answer_pos_train,self.answer_neg_train,self.label_neg_train, self.label_pos_train)
        
    def get_test_examples(self,data):
        question_test=[]
        answer_pos_test=[]
        answer_neg_test=[]
        label_pos_test=[]
        label_neg_test=[]
        value = data['triples']
        has_answer = False
        for triple in value:
          question_test.append(data['question'])
          answer_neg_test.append('centity '+triple['c_et']+' sentence '+triple['sentence']+' qentity '+triple['q_et'])
          #answer_neg_test.append(triple['c_et'])
          answer_pos_test.append(1 if triple['rank'] else 0)
          label_pos_test.append(triple['score'])
          label_neg_test.append(triple['score'])
          if triple['ans']==1:
            has_answer=True

        if has_answer==False:
          question_test=[]
          answer_pos_test=[]
          answer_neg_test=[]
        return self._create_examples(question_test,answer_pos_test,answer_neg_test,label_neg_test,label_pos_test)

    def _create_examples(self,question,answer_pos,answer_neg,label_neg,label_pos):
        examples = []
        for (i, (question,answer_pos,answer_neg,label_neg,label_pos)) in enumerate(zip(question,answer_pos,answer_neg,label_neg,label_pos)):
            examples.append(
                InputExample(question=question,answer_pos=answer_pos,answer_neg=answer_neg,label_neg=label_neg,label_pos=label_pos))
        return examples

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    
    features = []
    for (ex_index, example) in enumerate(examples):
        #-----------------------------question------------------------------

        question_input_ids = tokenizer.encode(example.question,add_special_tokens=True, max_length=max_seq_length,truncation=True)
        
        if len(question_input_ids) > max_seq_length:
            question_input_ids = question_input_ids[:(max_seq_length)]
        question_input_mask = [1] * len(question_input_ids)

        padding = [0] * (max_seq_length - len(question_input_ids))
        question_input_ids += padding
        question_input_mask += padding

        #-----------------------------answer_pos------------------------------

        answer_pos_input_ids = tokenizer.encode(example.answer_pos,add_special_tokens=True, max_length=max_seq_length,truncation=True)
        
        if len(answer_pos_input_ids) > max_seq_length:
            answer_pos_input_ids = answer_pos_input_ids[:(max_seq_length)]
        answer_pos_input_mask = [1] * len(answer_pos_input_ids)

        padding = [0] * (max_seq_length - len(answer_pos_input_ids))
        answer_pos_input_ids += padding
        answer_pos_input_mask += padding
        
        #-----------------------------answer_neg------------------------------

        answer_neg_input_ids = tokenizer.encode(example.answer_neg,add_special_tokens=True, max_length=max_seq_length,truncation=True)
        
        if len(answer_neg_input_ids) > max_seq_length:
            answer_neg_input_ids = answer_neg_input_ids[:(max_seq_length)]
        answer_neg_input_mask = [1] * len(answer_neg_input_ids)

        padding = [0] * (max_seq_length - len(answer_neg_input_ids))
        answer_neg_input_ids += padding
        answer_neg_input_mask += padding


        #--------------------------all to gether----------------------------
        input_ids=question_input_ids
        input_mask=question_input_mask

        output_pos_ids=answer_pos_input_ids
        output_pos_mask=answer_pos_input_mask
        output_neg_ids=answer_neg_input_ids
        output_neg_mask=answer_neg_input_mask
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              output_pos_ids=output_pos_ids,
                              output_pos_mask=output_pos_mask,
                              output_neg_ids=output_neg_ids,
                              output_neg_mask=output_neg_mask,
                              label_neg=example.label_neg,
                              label_pos=example.label_pos))
    return features


#files directory

test_dir = '../data/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic_test_scored.json'
train_dir = '../data/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic_train_scored.json'
output_test_file = '../data/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic_result.json'
f_out = open(output_test_file,'w')   

df_train = pd.read_json(train_dir, lines=True)


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


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_examples = DataProcessor().get_train_examples()

train_features = convert_examples_to_features(train_examples, MAX_SEQ_LENGTH, tokenizer)
del train_examples
gc.collect()

model = BertForSequenceRegression.from_pretrained('bert-base-uncased')
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
    
def my_loss(outputs_pos, outputs_neg):
  x,y= outputs_pos.shape
  zero = torch.zeros(x,y).to(device)
  one = torch.ones(x,y).to(device)
  loss = torch.max(zero,one-outputs_pos+outputs_neg)
  loss = torch.mean(loss)
  return loss

def train(model: nn.Module, num_epochs: int, learning_rate: float):
    num_train_optimization_steps = len(train_dataloader) * num_epochs 
    optimizer = get_optimizer(num_train_optimization_steps, learning_rate)
    assert all([x["lr"] == learning_rate for x in optimizer.param_groups])
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0  
    model.train()
    mb = master_bar(range(num_epochs))
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0    
    for _ in mb:
        for step, batch in enumerate(progress_bar(train_dataloader, parent=mb)):
            batch = tuple(t.to(device) for t in batch)
            b_all_input_ids,b_all_input_masks,b_all_output_pos_ids,b_all_output_pos_masks,b_all_output_neg_ids,b_all_output_neg_masks,label_neg,label_pos = batch
            pos = model(b_all_input_ids, attention_mask=b_all_input_masks,output_ids=b_all_output_pos_ids,output_attention_mask=b_all_output_pos_masks,label=label_pos)
            neg = model(b_all_input_ids, attention_mask=b_all_input_masks,output_ids=b_all_output_neg_ids,output_attention_mask=b_all_output_neg_masks,label=label_neg)
            loss = my_loss(pos,neg)
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
            nb_tr_examples += b_all_input_ids.size(0)
            nb_tr_steps += 1
            if FP16:
                lr_this_step = (
                     LR * warmup_linear(global_step/num_train_optimization_steps, WARMUP_PROPORTION))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            mb.child.comment = f'loss: {tr_loss:.4f} lr: {optimizer.get_lr()[0]:.2E}'
    logger.info("  train loss = %.4f", tr_loss) 
    f_out.write(" train loss = "+ str(tr_loss))
    return tr_loss
    
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_output_pos_ids = torch.tensor([f.output_pos_ids for f in train_features], dtype=torch.long)
all_output_pos_mask = torch.tensor([f.output_pos_mask for f in train_features], dtype=torch.long)
all_output_neg_ids = torch.tensor([f.output_neg_ids for f in train_features], dtype=torch.long)
all_output_neg_mask = torch.tensor([f.output_neg_mask for f in train_features], dtype=torch.long)
all_label_neg = torch.tensor([f.label_neg for f in train_features], dtype=torch.float)
all_label_pos = torch.tensor([f.label_pos for f in train_features], dtype=torch.float)
train_data = TensorDataset(all_input_ids, all_input_mask, all_output_pos_ids, all_output_pos_mask,all_output_neg_ids, all_output_neg_mask,all_label_neg,all_label_pos)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

set_trainable(model, True)
set_trainable(model.bert_a.embeddings, False)
set_trainable(model.bert_a.encoder, False)
set_trainable(model.bert_q.embeddings, False)
set_trainable(model.bert_q.encoder, False)
count_model_parameters(model)
train(model, num_epochs = 2, learning_rate = 5e-4)

# Save a trained model
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = '../model/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic-regressor.pth'
#torch.save(model_to_save.state_dict(), output_model_file)
#model.load_state_dict(torch.load(output_model_file))

#model.load_state_dict(torch.load(output_model_file))

gc.collect()

# Train the last two layer, too
set_trainable(model.bert_a.encoder.layer[11], True)
set_trainable(model.bert_a.encoder.layer[10], True)
set_trainable(model.bert_q.encoder.layer[11], True)
set_trainable(model.bert_q.encoder.layer[10], True)
count_model_parameters(model)
train(model, num_epochs = 2, learning_rate = 5e-5)

# Save a trained model
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = '../model/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic-regressor.pth'
#model.load_state_dict(torch.load(output_model_file))
torch.save(model_to_save.state_dict(), output_model_file)

# Train all layers
set_trainable(model, True)
count_model_parameters(model)
train(model, num_epochs = 1, learning_rate = 1e-5)

#Save a trained model
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = '../model/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic-regressor.pth'
torch.save(model_to_save.state_dict(), output_model_file)

# Save a trained model
#model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
#output_model_file = "./model/Qblink_50_bert_regressor.bin"
#torch.save(model_to_save.state_dict(), output_model_file)

# del train_features
gc.collect()

model = BertForSequenceRegression.from_pretrained('bert-base-uncased',cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
model.load_state_dict(torch.load('../model/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic-regressor.json'))
model.to(device)


model.eval()

counter=0


#test
class InputFeatures(object):

    def __init__(self, input_ids, input_mask,score,output_neg_ids,output_neg_mask,label_neg,label_pos):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.score = score
        self.output_neg_ids = output_neg_ids
        self.output_neg_mask = output_neg_mask
        self.label_neg=label_neg
        self.label_pos=label_pos

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    
    features = []
    for (ex_index, example) in enumerate(examples):
        #-----------------------------question------------------------------

        question_input_ids = tokenizer.encode(example.question,add_special_tokens=True, max_length=max_seq_length,truncation=True)
        
        if len(question_input_ids) > max_seq_length:
            question_input_ids = question_input_ids[:(max_seq_length)]
        question_input_mask = [1] * len(question_input_ids)

        padding = [0] * (max_seq_length - len(question_input_ids))
        question_input_ids += padding
        question_input_mask += padding
        
        #-----------------------------answer_neg------------------------------

        answer_neg_input_ids = tokenizer.encode(example.answer_neg,add_special_tokens=True, max_length=max_seq_length,truncation=True)
        
        if len(answer_neg_input_ids) > max_seq_length:
            answer_neg_input_ids = answer_neg_input_ids[:(max_seq_length)]
        answer_neg_input_mask = [1] * len(answer_neg_input_ids)

        padding = [0] * (max_seq_length - len(answer_neg_input_ids))
        answer_neg_input_ids += padding
        answer_neg_input_mask += padding


        #--------------------------all to gether----------------------------
        input_ids=question_input_ids
        input_mask=question_input_mask

        output_neg_ids=answer_neg_input_ids
        output_neg_mask=answer_neg_input_mask
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              score=example.answer_pos ,
                              output_neg_ids=output_neg_ids,
                              output_neg_mask=output_neg_mask,
                              label_neg=example.label_neg,
                              label_pos=example.label_pos))
    return features

BATCH_SIZE=120
counter=0
ap = 0
ap1=0
ndcg = 0
ndcg10 = 0
model.eval()
with open(test_dir, 'r') as f:
    for line in f:
        counter = counter + 1
        data=json.loads(line)
        question = data['question']
        indexx = data['index']
        test_examples = DataProcessor().get_test_examples(data)
        test_features = convert_examples_to_features(test_examples, MAX_SEQ_LENGTH, tokenizer)

        if len(test_examples)!=0:

          all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
          all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
          all_score = torch.tensor([f.score for f in test_features], dtype=torch.long)
          all_output_neg_ids = torch.tensor([f.output_neg_ids for f in test_features], dtype=torch.long)
          all_output_neg_mask = torch.tensor([f.output_neg_mask for f in test_features], dtype=torch.long)
          all_label_neg = torch.tensor([f.label_neg for f in test_features], dtype=torch.float)
          all_label_pos = torch.tensor([f.label_pos for f in test_features], dtype=torch.float)
          test_data = TensorDataset(all_input_ids, all_input_mask, all_score, all_output_neg_ids, all_output_neg_mask,all_label_neg,all_label_pos)
          test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
          
          input_ids = all_input_ids.to(device)
          input_mask = all_input_mask.to(device)
          score = all_score.to(device)
          all_output_neg_ids = all_output_neg_ids.to(device)
          all_output_neg_mask = all_output_neg_mask.to(device)
          label_pos=all_label_pos.to(device)
          label_neg=all_label_neg.to(device)

          with torch.no_grad():
            neg = model(input_ids, attention_mask=input_mask,output_ids=all_output_neg_ids,output_attention_mask=all_output_neg_mask,label=label_neg)
          
          
          neg = neg.cpu().detach().numpy()
          score = score.cpu().detach().numpy()
          label = label_neg.cpu().detach().numpy()
          score2 = []
          rank = 0
          for s in score:
              score2.append(1 if s>0 else 0)
          r = 1
          #added
          #lists =[]
          #for n,s,t,st in zip(neg,score,data["triples"],score2):
           #   lists.append((n,s,t,st))
          #lists.sort(key=lambda s:s[0],reverse=True)
          #lists = lists[:100]
          #score =[]
          #neg =[]
          #tripless =[]
          #score2=[]
          #for l in lists:
           #   neg.append(l[0])
            #  score.append(l[1])
             # tripless.append(l[2])
             # score2.append(l[3])
          #end




          for triple,ne in zip(data['triples'],neg):
              f_out.write(indexx+"\tQ0"+"\t"+"<dbpedia:"+triple["c_et"].replace(" ","_")+">"+"\t"+str(r)+"\t"+str(ne.item())+"\tCER"+"\n")  
              triple["ranking_score"]=ne.item()
              r=r+1
          #f_out.write(json.dumps(data)+"\n")
          score2= np.array(score2)
          #ap = ap + average_precision_score(score2, neg)
          ap = ap + average_precision_score(score2, neg)
          ap1 = ap1 + average_precision_score(score2, label)
          #print(neg)
          #print(label)
          neg = np.squeeze(neg)
          neg = np.asarray([neg])
          label = np.asarray([label])
          
          score = np.asarray([score])
          #print((score))
          #print((label))
          ndcg = ndcg+ndcg_score(score,neg,k=10 )
          ndcg10 = ndcg10+ndcg_score(score,neg,k=100 )
          f_out.write(indexx + "\t" + question + " ndcg: " + str(ndcg) + " ndcg10: " +str(ndcg10)+"\n")

print(ap/counter)
ap = float(ap)/counter
ap1 = float(ap1)/counter
ndcg = float(ndcg)/counter
ndcg10 = float(ndcg10)/counter

#f_out.write("map:"+ str(ap))
#f_out.write("map1:"+ str(ap1))
f_out.write("ndcg:"+ str(ndcg))
f_out.write("ndcg10:"+ str(ndcg10))
f_out.close()




        

