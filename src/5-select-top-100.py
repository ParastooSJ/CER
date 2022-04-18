import json
import unicodedata
import numpy as np
from sklearn.metrics import average_precision_score

train_path = './data/qblink-train-all-scored.json'
test_path ='./data/qblink-test-all-scored.json'

train = open(train_path,'r')
test = open(test_path,'r')

src_train_dev = [train,test]

train_dest_path = './data/qblink-train-5-ordered-bulk.json'
test_dest_path = './data/qblink-test-5-ordered-bulk.json'


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
    question = jobj["question"]
    answer = jobj["answer"]
    triples = jobj["triples"]
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
          tripless.append({"q_et":triple["q_et"],"rank":triple["rank"],"sentence":triple["sentence"],"c_et":triple["c_et"],"ans":triple["ans"],"score":triple["score"]})
          labels.append(0 if triple["rank"]==0 else 1)
          scores.append(triple["score"])
      labels = np.array(labels)
      scores = np.array(scores)
      
    wj = {"question":question,"triples":tripless}
    dest_train_dev[index].write(json.dumps(wj)+"\n")
  index = index + 1

