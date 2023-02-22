import torch
import json
import random
from sentence_transformers import SentenceTransformer, util
import os

class RandomSelector:

    def __init__(self,input_dir,output_dir,device):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.device = device
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v1').to(self.device)
        self.weight = 0.5

    def sample(self):
        input_f = open(self.input_dir,'r')
        if os.path.isfile(self.output_dir)==False:
            output_f = open(self.output_dir,'w')
            for line in input_f:
                jline = json.loads(line)
                question = jline["question"]
                en_question = self.model.encode(question)
                candidates = jline["candidates"]
                for object in candidates.keys():
                    rank = candidates[object]["rank"]
                    ans = candidates[object]["ans"]
                    rel_sim = -1 if rank==0 else float(rank)
            
                    for subject in candidates[object]["subject"].keys():
                        relation = candidates[object]["subject"][subject][0]
                        en_relation_object = self.model.encode(relation+" "+object)
                        relation_object_score = (1-self.weight) * rel_sim + self.weight * float(util.pytorch_cos_sim(en_question, en_relation_object)[0][0])
                        candidates[object]["relation_object_score"] = relation_object_score
                jline["candidates"] = candidates
                output_f.write(json.dumps(jline)+"\n")
            output_f.close() 