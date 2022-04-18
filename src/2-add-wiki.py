from gensim import utils
import json
import random
import sys
import ast

if len(sys.argv)!=2:
    print("enter type : (train - dev - test) as argument.")
    exit()
else:
    type=sys.argv[1]
    inputdata='./data/3-'+type+'.json'
    outputdata='./data/qblink_'+type+'_with_glossery.json'
    


bert_dataset = open(outputdata,'w', encoding='utf-8')


 
with utils.open(inputdata, 'rb') as f:
    for line in f:
        try:
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



