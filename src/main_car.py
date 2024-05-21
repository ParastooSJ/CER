from transformers import EncoderDecoderModel, BertTokenizer,AdamW,get_scheduler
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertConfig

from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule, SCHEDULES
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_percentage_error
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score
from sklearn.metrics import ndcg_score
import sys
import re
import gc
import logging
from train import *
from test import *
from Model.CR import *
from Model.CERCONTEXT import *
from Model.CERENTS import *
from RandomSelector import *
from fastprogress import master_bar, progress_bar



class CreateCer():
    def __init__(self, dataset, category, model_type, device, cache_dir):
        
        self.dataset = dataset
        self.category = category
        self.model_type = model_type
        self.device = device
        self.cache_dir = cache_dir

        self.train_dir = "../new_data/"+dataset+"/fold_"+category+"/train.json"
        self.test_dir = "../new_data/"+dataset+"/fold_"+category+"/test.json"
        self.sampled_train_dir = "../new_data/"+dataset+"/fold_"+category+"/sampled_train.json"
        self.scored_train_dir = "../new_data/"+dataset+"/fold_"+category+"/scored_train.json"
        self.scored_test_dir = "../new_data/"+dataset+"/fold_"+category+"/scored_test.json"
        self.final_dir = "../new_data/"+dataset+"/fold_"+category+"/"+model_type+"_final.txt"

        self.CRmodel_dir =  "../new_model/"+dataset+"/fold_"+category+"/CR.pth"
        self.CERmodel_dir =  "../new_model/"+dataset+"/fold_"+category+"/"+model_type+".pth"

        self.CR_model = CR.from_pretrained('bert-base-uncased',cache_dir=cache_dir).to(device)
        if model_type == "CERENTS": 
            self.CER_model = CERENTS.from_pretrained('bert-base-uncased',cache_dir=cache_dir).to(device)
        else:
            self.CER_model = CERCONTEXT.from_pretrained('bert-base-uncased',cache_dir=cache_dir).to(device)
        
        self.train_batch_size = 16
        self.test_batch_size = 200


    def random_select(self):
        rs = RandomSelector(input_dir=self.train_dir,output_dir=self.sampled_train_dir,device=self.device)
        rs.sample()

    def train_CR_model(self):
        if os.path.isfile(self.CRmodel_dir)==False:
            train_step(model_type="CR",model=self.CR_model,model_dir=self.CRmodel_dir,train_dir=self.sampled_train_dir,batch_size=self.train_batch_size,device=self.device)


    def test_CR_model(self):
        if os.path.isfile(self.scored_test_dir)==False:
            self.CR_model.load_state_dict(torch.load(self.CRmodel_dir))
            self.CR_model.to(self.device)
            test_score_step(model_type="CR",model=self.CR_model,test_dir=self.test_dir,train_dir=self.train_dir,
                output_test_dir=self.scored_test_dir,output_train_dir=self.scored_train_dir,batch_size=self.test_batch_size,device=self.device)
        

    def test_CER_model(self):
        test_step(model_type=self.model_type,model=self.CR_model,input_dir=self.scored_test_dir,output_dir=self.final_dir,batch_size=self.test_batch_size,device=self.device)



def main():
    dataset = sys.argv[1]
    category = sys.argv[2]
    model_type = sys.argv[3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = "../cache"

    cc = CreateCer(dataset=dataset, category=category, model_type=model_type, device=device, cache_dir=cache_dir)
    cc.random_select()
    
    
    cc.train_CR_model()
    cc.test_CR_model()
    cc.test_CER_model()
    
if __name__ == '__main__':
    main()
    
