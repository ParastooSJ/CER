
import torch
import torch.nn as nn
import json
from tqdm.auto import tqdm
import logging
from Loader import *
from fastprogress import master_bar, progress_bar
#import progress_bar

def get_score(model,test_dataloader,model_type,device):
    
    model.eval()
    counter=0
    print()
    mb = tqdm(test_dataloader)
    #mb = progress_bar(test_dataloader)
    for batch in mb:
        batch = tuple(t.to(device) for t in batch)
        if model_type == "CR":
            input_ids, input_mask, segment_ids = batch
            with torch.no_grad():
                outputs = model(input_ids, segment_ids, input_mask)
        elif model_type=="CERENTS":
            input_ids,input_mask,output_ids,output_mask,entity_no,label = batch
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=input_mask,output_ids=output_ids,output_attention_mask=output_mask,entity_no=entity_no,label=label_neg)
        else:
            input_ids,input_mask,output_ids,output_mask,label = batch
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=input_mask,output_ids=output_ids,output_attention_mask=output_mask,label=label)

        outputs = outputs.detach().cpu().numpy()
        print('-------')
        print(outputs)
        if counter==0:
            all_outputs=torch.from_numpy(outputs)
            counter+=1
        else:
            all_outputs=torch.cat([all_outputs, torch.from_numpy(outputs)], dim=0)
            
    return all_outputs





def test_score_step(model_type,model,test_dir,train_dir,output_test_dir,output_train_dir,batch_size,device):

    train_f = open(train_dir,'r')
    test_f = open(test_dir,'r')

    scored_train_f = open(output_train_dir,'w')
    scored_test_f = open(output_test_dir,'w')

    input_f = [train_f,test_f]
    output_f = [scored_train_f,scored_test_f]

    counter = 0
    for i in range(0,len(input_f)):
        for line in input_f[i]:
            data=json.loads(line)
            counter+=1
            
            if len(data['candidates']) > 0:
                
                test_dataloader = test_data_loader(model_type,data,batch_size)
                
        
                scores = get_score(model,test_dataloader,model_type,device)
                
                lists = []
                for object,score in zip(data["candidates"].keys(),scores):
                    data['candidates'][object]["prune_score"]=score.item()
                    lists.append((object,score.item()))
                lists.sort(key=lambda s:s[1], reverse=True)
                lists = lists[:1000]
                keep_candidates = []
                for l in lists:
                    keep_candidates.append(l[0])
                new_candidates = {}
                for object in keep_candidates:
                    new_candidates[object] = data['candidates'][object]
                data['candidates'] = new_candidates
                output_f[i].write(json.dumps(data)+"\n")
        output_f[i].close()


def test_step(model_type,model,input_dir,output_dir,batch_size,device):
    
    test_f = open(input_dir,'r')
    scored_test_f = open(output_dir,'w')
    
    model.eval()
    for line in test_f:
        object_list = set()
        data=json.loads(line)
        question = data['question']
        indexx = data['index']
        
        
        if len(data['candidates'])!=0:
            
           

            lists = []
            for object in data["candidates"].keys():
                lists.append((object,data['candidates'][object]["prune_score"]))
            lists.sort(key=lambda s:s[1], reverse=True)
            lists = lists[:1000]
            r = 0
            ind = data["index"].replace(" ","_")
            
            for l in lists:
                if l[0].replace(" ","_").lower() not in object_list:
                    
                   
                        scored_test_f.write(ind.lower()+"\tQ0"+"\t"+l[0].replace(" ","_").lower()+"\t"+str(r)+"\t"+str(l[1])+"\tCER"+"\n")
                        object_list.add(l[0].replace(" ","_").lower())
                        r += 1

        
            print('hree')

    scored_test_f.close()
