from gensim import utils
import json
import random
import sys
import ast

if len(sys.argv)==3:

    inputdata="../data/"+sys.argv[1]+"/"+sys.argv[1]+"_fold_"+sys.argv[2]+"/train.json"
    outputdata="../data/"+sys.argv[1]+"/"+sys.argv[1]+"_fold_"+sys.argv[2]+"/basic_random_train.json"
    

bert_dataset = open(outputdata,'w', encoding='utf-8')

dataset_for_learn=[]   
with utils.open(inputdata, 'rb') as f:
    for line in f:
        try:
            question = json.loads(line)
            triples = question["triples"]
            index = question["index"]
            q_ets = question["q_ets"]
            conf = question["conf"]
            all_selected_entities = set()
            selected_triples=[]
            negss = []
            neg_set=set()
            for triple in triples:
                if triple["ans"]:
                    if triple["c_et"] not in all_selected_entities:
                        selected_triples.append(triple)
                        triples.remove(triple)
                        all_selected_entities.add(triple["c_et"])
                else:
                    if triple['c_et'] not in neg_set:
                        neg_set.add(triple["c_et"])
                        negss.append(triple)
            num =300
            if num > len(negss):
                num = len(negss)
            negetive_triples=random.sample(negss, num)
            negs =[]
            for neg in negetive_triples:
                if neg["c_et"] not in all_selected_entities and len(negs)<100:
                #if len(negs)<100:
                    negs.append(neg)
                    all_selected_entities.add(neg["c_et"])
            selected_triples.extend(negs)

            for triple in selected_triples:

                
                jsonstr={"index":question["index"],"question":question["question"],"q_ets":question["q_ets"],"conf":question["conf"],"triple":triple}
                
                dataset_for_learn.append(jsonstr)
                
        except Exception as e:
            
            print(e)
            pass
        
json.dump(dataset_for_learn,bert_dataset)
bert_dataset.close()




