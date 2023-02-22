from transformers import BertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import gc

import DataProcessor.DataProcessorCerContext as context
import DataProcessor.DataProcessorCerDis as dis
import DataProcessor.DataProcessorCerEnts as ents
import DataProcessor.DataProcessorCR as cr



def train_data_loader(model_type,train_data,batch_size):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    MAX_SEQ_LENGTH = 100
    if model_type == "CR":
        train_examples = cr.DataProcessorCR().get_train_examples(train_data)
        train_features = cr.convert_examples_to_features_cr(train_examples, MAX_SEQ_LENGTH, tokenizer)
        del train_examples
        gc.collect()
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_scores = torch.tensor([f.score for f in train_features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_scores)
        train_sampler = RandomSampler(train_data)


    elif model_type == "CERENTS":
        train_examples = ents.DataProcessorCerEnts().get_train_examples(train_data)
        train_features = ents.convert_examples_to_features_cer_ents(train_examples, MAX_SEQ_LENGTH, tokenizer)
        del train_examples
        gc.collect()
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_output_pos_ids = torch.tensor([f.output_pos_ids for f in train_features], dtype=torch.long)
        all_output_pos_mask = torch.tensor([f.output_pos_mask for f in train_features], dtype=torch.long)
        all_output_neg_ids = torch.tensor([f.output_neg_ids for f in train_features], dtype=torch.long)
        all_output_neg_mask = torch.tensor([f.output_neg_mask for f in train_features], dtype=torch.long)
        all_entity_no = torch.tensor([f.entity_no for f in train_features], dtype=torch.long)
        all_label_neg = torch.tensor([f.label_neg for f in train_features], dtype=torch.float)
        all_label_pos = torch.tensor([f.label_pos for f in train_features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_output_pos_ids, all_output_pos_mask,all_output_neg_ids, all_output_neg_mask,all_entity_no,all_label_neg,all_label_pos)
        train_sampler = RandomSampler(train_data)
    else:
        if model_type =="CERCONTEXT":
            train_examples = context.DataProcessorCerContext().get_train_examples(train_data)
            train_features = context.convert_examples_to_features_cer_context(train_examples, MAX_SEQ_LENGTH, tokenizer)
        else:
            train_examples = dis.DataProcessorCerDis().get_train_examples(train_data)
            train_features = dis.convert_examples_to_features_cer_dis(train_examples, MAX_SEQ_LENGTH, tokenizer)
        del train_examples
        gc.collect()
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_output_pos_ids = torch.tensor([f.output_pos_ids for f in train_features], dtype=torch.long)
        all_output_pos_mask = torch.tensor([f.output_pos_mask for f in train_features], dtype=torch.long)
        all_output_neg_ids = torch.tensor([f.output_neg_ids for f in train_features], dtype=torch.long)
        all_output_neg_mask = torch.tensor([f.output_neg_mask for f in train_features], dtype=torch.long)
        all_label_neg = torch.tensor([f.label_neg for f in train_features], dtype=torch.float)
        all_label_pos = torch.tensor([f.label_pos for f in train_features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_output_pos_ids, all_output_pos_mask,all_output_neg_ids, all_output_neg_mask,all_label_neg,all_label_pos)
        train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader



def test_data_loader(model_type,data,batch_size):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    MAX_SEQ_LENGTH = 100
    if model_type == "CR":
        test_examples = cr.DataProcessorCR().get_test_examples(data)
        test_features = cr.convert_examples_to_features_cr(test_examples, MAX_SEQ_LENGTH, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        test_sampler = SequentialSampler(test_data)
        

    elif model_type =="CERENTS":
        test_examples = ents.DataProcessorCerEnts().get_test_examples(data)
        test_features = ents.convert_examples_to_features_cer_ents(test_examples, MAX_SEQ_LENGTH, tokenizer)
        
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_output_ids = torch.tensor([f.output_neg_ids for f in test_features], dtype=torch.long)
        all_output_mask = torch.tensor([f.output_neg_mask for f in test_features], dtype=torch.long)
        all_entity_no = torch.tensor([f.entity_no for f in test_features], dtype=torch.long)
        all_label_neg = torch.tensor([f.label_neg for f in test_features], dtype=torch.float)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_output_ids, all_output_mask,all_entity_no,all_label_neg)
        

    else:
        if model_type =="CERCONTEXT":
            test_examples = context.DataProcessorCerContext().get_test_examples(data)
            test_features = context.convert_examples_to_features_cer_context(test_examples, MAX_SEQ_LENGTH, tokenizer)
        else:
            test_examples = dis.DataProcessorCerDis().get_test_examples(data)
            test_features = dis.convert_examples_to_features_cer_ents(test_examples, MAX_SEQ_LENGTH, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_output_ids = torch.tensor([f.output_neg_ids for f in test_features], dtype=torch.long)
        all_output_mask = torch.tensor([f.output_neg_mask for f in test_features], dtype=torch.long)
        all_label = torch.tensor([f.label_neg for f in test_features], dtype=torch.float)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_output_ids, all_output_mask,all_label)
    print(len(test_data))
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler,batch_size=batch_size)
    return test_dataloader

