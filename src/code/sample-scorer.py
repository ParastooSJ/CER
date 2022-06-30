import json
#from nltk import tokenize
from collections import OrderedDict
#import nltk
from operator import itemgetter
#nltk.download('punkt')
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import sys
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('paraphrase-distilroberta-base-v1').to(device)
weight = 0.5

if len(sys.argv)==4:


    train_path = '../data/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic_random_'+sys.argv[3]+'.json'
    #test_path = './data/4-test.json' 

    train_path_w = '../data/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic_random_'+sys.argv[3]+'_scored.json'


    train_f = open(train_path,'r') 
    #test_f = open(test_path,'r')


    train_f_w = open(train_path_w,'w') 
    #test_f_w = open(test_path_w,'w')
    write_f = [train_f_w]

    train = json.load(train_f)
    #test = json.load(test_f)
    read_f = [train]

    train_cont = False
    index = 0
    count = 0
    for s_f in read_f :
        output = []
        can_go = True
        for line in s_f:
            if index == 0 :
                count = count+1
            if count <=0 and index ==0:
                can_go = False
            else:
                can_go = True     
     
            try:
                if can_go:
                    question = line['question']
                    index2 = line['index']
                    sentence = line['triple']['sentence']
                    c_et = line['triple']['c_et']
                    q_et = line['triple']['q_et'] 
                    answer = line['triple']['ans']
                    rank = line['triple']['rank']
                    en_sent = model.encode(c_et)
                    en_question = model.encode(question)
                    cos_sim = util.pytorch_cos_sim(en_question, en_sent)
                    cos_sim_float = float(cos_sim[0][0])
                    rel_sim = -1 if rank==0 else 1
                    score = weight * cos_sim_float + (1-weight) * rel_sim
                    line['triple']['score'] = score
                    jsonobj = {"index":index2,"question":question,"q_ets":line["q_ets"],"conf":line["conf"],"triple":line['triple']}
                    ss = json.dumps(jsonobj)
                    write_f[index].write(json.dumps(jsonobj)+"\n")
            except Exception as e:
                print(e)
        
  #json.dump(output,write_f[index])
    index = index+1
  #write_f[index].close()












