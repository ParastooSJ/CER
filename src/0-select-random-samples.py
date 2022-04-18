from gensim import utils
import json
import random
import sys
import ast

if len(sys.argv)==2:

    inputdata="./data/3-"+sys.argv[1]+".json"
    outputdata="./data/4-"+sys.argv[1]+".json"
    

bert_dataset = open(outputdata,'w', encoding='utf-8')

dataset_for_learn=[]   
with utils.open(inputdata, 'rb') as f:
    for line in f:
        try:
            question = json.loads(line)
            triples = question["triples"]
            index = question["index"]
            all_selected_entities = set()
            selected_triples=[]
            for triple in triples:
                if triple["ans"]:
                    if triple["c_et"] not in all_selected_entities:
                        selected_triples.append(triple)
                        triples.remove(triple)
                        all_selected_entities.add(triple["c_et"])
            num =300
            if num > len(triples):
                num = len(triples)
            negetive_triples=random.sample(triples, num)
            negs =[]
            for neg in negetive_triples:
                if neg["c_et"] not in all_selected_entities and len(negs)<100:
                    negs.append(neg)
                    all_selected_entities.add(neg["c_et"])
            selected_triples.extend(negs)

            for triple in selected_triples:

                
                jsonstr={"index":question["index"],"question":question["question"],"triple":{"q_et":triple["q_et"],"c_et":triple["c_et"],"sentence":triple["sentence"],"ans":triple["ans"],"rank":triple["rank"]}}
                
                dataset_for_learn.append(jsonstr)
                
        except Exception as e:
            
            print(e)
            pass
        
json.dump(dataset_for_learn,bert_dataset)
bert_dataset.close()




