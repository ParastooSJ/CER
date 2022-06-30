import json
import unicodedata
import numpy as np
import sys
from sklearn.metrics import average_precision_score

train_path = '../data/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic_train_scored.json'
test_path ='../data/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic_test_scored.json'

train = open(train_path,'r')
test = open(test_path,'r')

src_train_dev = [train,test]

train_dest_path = '../data/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic_train_ordered.json'
test_dest_path = '../data/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic_test_ordered.json'


train_dest = open(train_dest_path,'w')
test_dest = open(test_dest_path,'w')

neg_samples =100
dest_train_dev = [train_dest,test_dest]

def normalize(text):
  return unicodedata.normalize('NFD', text).replace('_', ' ').lower()

index = 0
for jfile in src_train_dev:
  id = 0
  while True:
    id = id+1
    line= jfile.readline()
    if not line:
      break
    jfinal = {}
    jobj = json.loads(line)
    indexx = jobj["index"]
    question = jobj["question"]
    answer = jobj["answer"]
    triples = jobj["triples"]
    conf = jobj["conf"]
    q_ets = jobj["q_ets"]
    chosen_triples = sorted(triples, key=lambda k: k["score"], reverse=True)[:100]
    jfinal["question"] = question
    jfinal["answer"] = answer
    availableAnswer = False
    for triple in chosen_triples:
        try:  
            if triple["ans"]:
                answer = normalize(triple["c_et"])
                availableAnswer = True
        except e as Exception:
            print("ex")
            break
    if availableAnswer:
      tripless = []
      labels = []
      scores = []
      for triple in chosen_triples:
          tripless.append(triple)
          labels.append(0 if triple["rank"]==0 else 1)
          scores.append(triple["score"])
      labels = np.array(labels)
      scores = np.array(scores)
    wj = {"index":indexx,"question":question,"q_ets":q_ets,"conf":conf,"triples":tripless}
    dest_train_dev[index].write(json.dumps(wj)+"\n")
  index = index + 1


