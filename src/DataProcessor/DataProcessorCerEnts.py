
import torch
import torch.nn as nn

class InputExampleCerEnts(object):
    def __init__(self,question,answer_neg,label_neg,answer_pos=None,label_pos=None):
        self.question=question
        self.answer_pos=answer_pos
        self.answer_neg = answer_neg
        self.label_neg = label_neg
        self.label_pos = label_pos

class InputFeaturesCerEnts(object):

    def __init__(self, input_ids, input_mask,output_neg_ids,output_neg_mask,entity_no,label_neg,output_pos_ids=None,output_pos_mask=None,label_pos=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.output_pos_ids = output_pos_ids
        self.output_pos_mask = output_pos_mask
        self.output_neg_ids = output_neg_ids
        self.output_neg_mask = output_neg_mask
        self.entity_no = entity_no
        self.label_neg = label_neg
        self.label_pos = label_pos

class DataProcessorCerEnts:

    def __init__(self):
        self.question_train=[]
        self.answer_pos_train=[]
        self.answer_neg_train=[]
        self.label_pos_train=[]
        self.label_neg_train=[]
    
    def get_train_examples(self,df_train):

        
        for id,value in df_train.iterrows():
            
            pos_answer= []
            pos_rank =[]
          
            _ , q_ets = get_entities(value['question'],0.001)
            for object in value['candidates'].keys():
                
                rank = value['candidates'][object]["rank"]
                prune_score = value['candidates'][object]["prune_score"]
                relation_object_score = value['candidates'][object]["relation_object_score"]

                for subject in value['candidates'][object]["subject"]:
                    relation = value['candidates'][object]["subject"][subject][0]
                    if rank>=1:
                        pos_answer.append('centity '+object+' sentence '+relation+' qentity '+subject)
                        pos_rank.append(prune_score)
                    
            if len(pos_answer)>0 :
                i = 0
                for pos in pos_answer:
                    for object in value['candidates']:
                        rank = value['candidates'][object]["rank"]
                        prune_score = value['candidates'][object]["prune_score"]
                        relation_object_score = value['candidates'][object]["relation_object_score"]

                        for subject in value['candidates'][object]["subject"]:
                            relation = value['candidates'][object]["subject"][subject][0]
                            if value['candidates'][object]['ans']==False:
                                self.question_train.append(q_ets)
                                self.answer_neg_train.append('centity '+object+' sentence '+relation+' qentity '+subject)
                                self.answer_pos_train.append(pos)
                                self.label_neg_train.append(prune_score)
                                self.label_pos_train.append(pos_rank[i])
                    i += 1
                


        return self._create_examples(self.question_train,self.answer_neg_train,self.label_neg_train, self.label_pos_train,self.answer_pos_train)
        
    def get_test_examples(self,data):
        question_test=[]
        answer_neg_test=[]
        label_neg_test=[]
        
        has_answer = False

        _ , q_ets = get_entities(data['question'],0.001)
        for object in data['candidates']:
            rank = data['candidates'][object]["rank"]
            prune_score = data['candidates'][object]["prune_score"]
            relation_object_score = data['candidates'][object]["relation_object_score"]

            for subject in data['candidates'][object]["subject"]:
                relation = data['candidates'][object]["subject"][subject][0]

                question_test.append(q_ets)
                answer_neg_test.append('centity '+object+' sentence '+relation+' qentity '+subject)
        
                label_neg_test.append(prune_score)
                if data['candidates'][object]["ans"]:
                    has_answer=True

        if has_answer==False:

            question_test=[]
        return self._create_examples(question_test,answer_neg_test,label_neg_test,None,None)

    def _create_examples(self,questio,answer_neg,label_neg,answer_pos,label_pos):
        examples = []
        if answer_pos ==None and label_pos==None:
            for (i, (question,answer_neg,label_neg)) in enumerate(zip(question,answer_neg,label_neg)):
                examples.append(
                    InputExampleCerContext(question=question,answer_neg=answer_neg,label_neg=label_neg,answer_pos=None,label_pos=None))
        else:
            for (i, (question,answer_neg,label_neg,answer_pos,label_pos)) in enumerate(zip(question,answer_neg,label_neg,answer_pos,label_pos)):
                examples.append(
                InputExampleCerEnts(question=question,answer_neg=answer_neg,label_neg=label_neg,answer_pos=answer_pos,label_pos=label_pos))
        return examples

def convert_examples_to_features_cer_ents(examples, max_seq_length, tokenizer):
    
    features = []
    for (ex_index, example) in enumerate(examples):
        #-----------------------------question------------------------------
        question_input_ids = []
        question_input_mask = []
        entity_no = len(example.question)
        for q_et in example.question:

            q_et_input_ids = tokenizer.encode(q_et,add_special_tokens=True, max_length=max_seq_length,truncation=True)
        
            if len(q_et_input_ids) > max_seq_length:
                q_et_input_ids = q_et_input_ids[:(max_seq_length)]
            q_et_input_mask = [1] * len(q_et_input_ids)

            padding = [0] * (max_seq_length - len(q_et_input_ids))
            q_et_input_ids += padding
            q_et_input_mask += padding
            question_input_ids.append(q_et_input_ids)
            question_input_mask.append(q_et_input_mask)
        while len(question_input_ids) < q_ets_size:
            padding = [0] * (max_seq_length )
            q_et_input_ids = padding
            q_et_input_mask = padding
            question_input_ids.append(q_et_input_ids)
            question_input_mask.append(q_et_input_mask)
        if len(question_input_ids) > q_ets_size:
            question_input_ids =  question_input_ids[:q_ets_size]
            question_input_mask = question_input_mask[:q_ets_size]
            entity_no = q_ets_size


        #-----------------------------answer_pos------------------------------
        if example.answer_pos !=None:

            answer_pos_input_ids = tokenizer.encode(example.answer_pos,add_special_tokens=True, max_length=max_seq_length,truncation=True)
            
            if len(answer_pos_input_ids) > max_seq_length:
                answer_pos_input_ids = answer_pos_input_ids[:(max_seq_length)]
            answer_pos_input_mask = [1] * len(answer_pos_input_ids)

            padding = [0] * (max_seq_length - len(answer_pos_input_ids))
            answer_pos_input_ids += padding
            answer_pos_input_mask += padding
        else:
            answer_pos_input_ids = None
            answer_pos_input_mask = None
        
        #-----------------------------answer_neg------------------------------

        answer_neg_input_ids = tokenizer.encode(example.answer_neg,add_special_tokens=True, max_length=max_seq_length,truncation=True)
        
        if len(answer_neg_input_ids) > max_seq_length:
            answer_neg_input_ids = answer_neg_input_ids[:(max_seq_length)]
        answer_neg_input_mask = [1] * len(answer_neg_input_ids)

        padding = [0] * (max_seq_length - len(answer_neg_input_ids))
        answer_neg_input_ids += padding
        answer_neg_input_mask += padding


        #--------------------------all to gether----------------------------
        input_ids=question_input_ids

        input_mask=question_input_mask

        output_pos_ids=answer_pos_input_ids
        output_pos_mask=answer_pos_input_mask
        output_neg_ids=answer_neg_input_ids
        output_neg_mask=answer_neg_input_mask
        features.append(
                InputFeaturesCerEnts(input_ids=input_ids,
                              input_mask=input_mask,
                              output_neg_ids=output_neg_ids,
                              output_neg_mask=output_neg_mask,
                              entity_no=entity_no,
                              label_neg=example.label_neg,
                              output_pos_ids=output_pos_ids,
                              output_pos_mask=output_pos_mask,
                              label_pos=example.label_pos))
    return features
