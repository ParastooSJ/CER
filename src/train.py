
import torch
import torch.nn as nn
import pandas as pd
from fastprogress import master_bar, progress_bar
from tqdm.auto import tqdm
import logging
import Loader
from ParameterPrep import *


def margin_loss(outputs_pos, outputs_neg):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  x,y= outputs_pos.shape
  zero = torch.zeros(x,y).to(device)
  one = torch.ones(x,y).to(device)
  loss = torch.max(zero,one-outputs_pos+outputs_neg)
  loss = torch.mean(loss)
  return loss


def train(model: nn.Module, num_epochs: int, learning_rate: float, train_dataloader, model_type, device):

    num_train_optimization_steps = len(train_dataloader) * num_epochs 
    optimizer = PrepareOptimizer(model).get_optimizer(num_train_optimization_steps, learning_rate)
    assert all([x["lr"] == learning_rate for x in optimizer.param_groups])
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0  
    n_gpu = torch.cuda.device_count()
    model.train()
    mb = master_bar(range(num_epochs))
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0    
    for _ in mb:
        for step, batch in enumerate(progress_bar(train_dataloader, parent=mb)):
            batch = tuple(t.to(device) for t in batch)
            if model_type == "CR":
                b_all_input_ids, input_mask, segment_ids, score = batch
                loss = model(b_all_input_ids, segment_ids, input_mask, score)

            elif model_type =="CERENTS":
                b_all_input_ids,b_all_input_masks,b_all_output_pos_ids,b_all_output_pos_masks,b_all_output_neg_ids,b_all_output_neg_masks,all_entity_no,label_neg,label_pos = batch
                pos = model(b_all_input_ids, attention_mask=b_all_input_masks,output_ids=b_all_output_pos_ids,output_attention_mask=b_all_output_pos_masks,entity_no=all_entity_no,label=label_pos)
                neg = model(b_all_input_ids, attention_mask=b_all_input_masks,output_ids=b_all_output_neg_ids,output_attention_mask=b_all_output_neg_masks,entity_no=all_entity_no,label=label_neg)
                loss = margin_loss(pos,neg)
                
            else:
                b_all_input_ids,b_all_input_masks,b_all_output_pos_ids,b_all_output_pos_masks,b_all_output_neg_ids,b_all_output_neg_masks,label_neg,label_pos = batch
                pos = model(b_all_input_ids, attention_mask=b_all_input_masks,output_ids=b_all_output_pos_ids,output_attention_mask=b_all_output_pos_masks,label=label_pos)
                neg = model(b_all_input_ids, attention_mask=b_all_input_masks,output_ids=b_all_output_neg_ids,output_attention_mask=b_all_output_neg_masks,label=label_neg)
                loss = margin_loss(pos,neg)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            
            loss.backward()

            if tr_loss == 0:
                tr_loss = loss.item()
            else:
                tr_loss = tr_loss * 0.9 + loss.item() * 0.1
            nb_tr_examples += b_all_input_ids.size(0)
            nb_tr_steps += 1
            
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            mb.child.comment = f'loss: {tr_loss:.4f} lr: {optimizer.get_lr()[0]:.2E}'
    logger.info("  train loss = %.4f", tr_loss) 
    return tr_loss


def train_step(model_type,model,model_dir,train_dir,batch_size,device):
    train_data = pd.read_json(train_dir, lines=True)
    train_dataloader = Loader.train_data_loader(model_type,train_data,batch_size)

    # Train the model without Bert layers
    set_trainable(model, True)
    if model_type =="CR":
        set_trainable(model.bert.embeddings, False)
        set_trainable(model.bert.encoder, False)
    else:
        set_trainable(model.bert_a.embeddings, False)
        set_trainable(model.bert_a.encoder, False)
        set_trainable(model.bert_q.embeddings, False)
        set_trainable(model.bert_q.encoder, False)
    count_model_parameters(model)
    train(model=model, num_epochs = 2, learning_rate = 5e-4,train_dataloader=train_dataloader,model_type=model_type,device=device)

    # Train the last two layer, too
    if model_type =="CR":
        set_trainable(model.bert.encoder.layer[11], True)
        set_trainable(model.bert.encoder.layer[10], True)
    else:
        set_trainable(model.bert_a.encoder.layer[11], True)
        set_trainable(model.bert_a.encoder.layer[10], True)
        set_trainable(model.bert_q.encoder.layer[11], True)
        set_trainable(model.bert_q.encoder.layer[10], True)
    count_model_parameters(model)
    train(model=model, num_epochs = 2, learning_rate = 5e-5,train_dataloader=train_dataloader,model_type=model_type,device=device)
    
    # Train all layers
    set_trainable(model, True)
    count_model_parameters(model)
    train(model=model, num_epochs = 1, learning_rate = 1e-5,train_dataloader=train_dataloader,model_type=model_type,device=device)

    #Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    torch.save(model_to_save.state_dict(), model_dir)
