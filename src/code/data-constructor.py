from gensim import utils
import json
import random
import sys
import ast

if len(sys.argv)!=4:
    print(len(sys.argv))
    print("enter type : (train - dev - test) as argument.")
    exit()
else:
    inputdata='../data/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/'+sys.argv[3]+'.json'
    outputdata='../data/'+sys.argv[1]+'/'+sys.argv[1]+'_fold_'+sys.argv[2]+'/basic_'+sys.argv[3]+'_with_wiki.json'
    print("in")


bert_dataset = open(outputdata,'w', encoding='utf-8')


 
with utils.open(inputdata, 'rb') as f:
    for line in f:
        try:
            #print(line)
            question = json.loads(line)
            triples=question["triples"]
            all_c_ets =set()
            final_triples = []
            for triple in triples:
                if triple["c_et"] not in all_c_ets:
                    all_c_ets.add(triple["c_et"])
                    final_triples.append(triple)
            question["triples"] = final_triples
            print(len(final_triples))
            bert_dataset.write(json.dumps(question)+"\n")
            # bert_dataset.write("\n")

        except Exception as e:
            
            print(e)
            pass
        
bert_dataset.close()





















