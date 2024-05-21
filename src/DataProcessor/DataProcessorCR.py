import torch
import torch.nn as nn
import numpy as np

class InputExampleCR(object):

    def __init__(self, question,relation,object,score=None):
        self.question=question
        self.relation=relation
        self.object=object
        self.score=score
    
 

class InputFeaturesCR(object):

    def __init__(self, input_ids, input_mask, segment_ids , score):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.score = score


class DataProcessorCR:
    
    def __init__(self):
        self.question_train=[]
        self.relation_train=[]
        self.object_train=[]
        self.score_train=[]

    def get_train_examples(self,df_train):
        

        for _,value in df_train.iterrows():
            question = value['question'][:1000].strip()
            for object in value['candidates'].keys():
                
                relation_object_score = value['candidates'][object]["relation_object_score"]
                #object_score = value['candidates'][object]["object_score"]
                for subject in value['candidates'][object]["subject"]:
                    relation  = value['candidates'][object]["subject"][subject][0]
                    self.question_train.append(question)
                    self.relation_train.append(relation[:1000].strip())

                    self.object_train.append(relation[:1000].strip()+" "+object[:1000].strip())
                    self.score_train.append(relation_object_score)

        self.score_train=(self.score_train-np.min(self.score_train))/(np.max(self.score_train)-np.min(self.score_train))
        return self._create_examples(self.question_train,self.relation_train,self.object_train,self.score_train)


    def get_test_examples(self,df_test):
        question_test=[]
        relation_test=[]
        object_test=[]
        score_test=[]
        #try:
        if True:
            question = df_test['question'][:1000].strip()
            for object in df_test['candidates'].keys():
                
                relation_object_score = 0 #df_test['candidates'][object]["relation_object_score"]
                #object_score = df_test['candidates'][object]["object_score"]
                for subject in df_test['candidates'][object]["subject"]:
                    relation  = df_test['candidates'][object]["subject"][subject][0]
                    question_test.append(question)
                    relation_test.append(relation[:1000].strip())

                    object_test.append(relation[:1000].strip()+" "+object[:1000].strip())
                    score_test.append(relation_object_score)
            
            score_test=(score_test-np.min(score_test))/(np.max(score_test)-np.min(score_test))
        #except:
            print("error")
        return self._create_examples(question_test,relation_test,object_test,score_test)

    def _create_examples(self,question,relation,object,score):
        examples = []
        for (i, (question,relation,object,score)) in enumerate(zip(question,relation,object,score)):
            examples.append(
                InputExampleCR(question=question,relation=relation,object=object,score=score))
        return examples

def convert_examples_to_features_cr(examples, max_seq_length, tokenizer):
    
    features = []
    for (ex_index, example) in enumerate(examples):
        question_input_ids = tokenizer.encode(example.question,add_special_tokens=True, max_length=max_seq_length,truncation=True)
        if len(question_input_ids) > max_seq_length:
            question_input_ids = question_input_ids[:(max_seq_length )]


        question_segment_ids = [0] * len(question_input_ids)
        question_input_mask = [1] * len(question_input_ids)

        padding = [0] * (max_seq_length - len(question_input_ids))
        question_input_ids += padding
        question_input_mask += padding
        question_segment_ids += padding

        #--------------------------object----------------------------

        object_input_ids = tokenizer.encode(example.object,add_special_tokens=True, max_length=max_seq_length,truncation=True)[1:]
        if len(object_input_ids) > max_seq_length :
            object_input_ids = object_input_ids[:(max_seq_length)]

        object_segment_ids = [1] * len(object_input_ids)
        object_input_mask = [1] * len(object_input_ids)

        padding = [0] * (max_seq_length - len(object_input_ids))
        object_input_ids += padding
        object_input_mask += padding
        object_segment_ids += padding

        #--------------------------all to gether----------------------------
        input_ids=question_input_ids+object_input_ids
        input_mask=question_input_mask+object_input_mask
        segment_ids=question_segment_ids+object_segment_ids
        features.append(
                InputFeaturesCR(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              score=example.score))
    return features
