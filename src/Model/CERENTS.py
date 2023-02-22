from transformers import BertModel as bm
import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel


class CERENTS(BertPreTrainedModel):
    def __init__(self,config):
        super(CERENTS, self).__init__(config)
        self.bert_q = bm.from_pretrained('bert-base-uncased')
        self.bert_a = bm.from_pretrained('bert-base-uncased')
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.linear = nn.Linear(2, 1)
       
    def forward(self, input_ids, attention_mask,output_ids,output_attention_mask,entity_no,label):
        a_output = self.bert_a(output_ids,output_attention_mask).pooler_output
        x , y = output_ids.shape
        summ = torch.zeros(x).to(device)
        
        for ind in range(0,q_ets_size):
            i = input_ids[:,ind,:]
            a = attention_mask[:,ind,:]
            q_e_output = self.bert_q(i,a).pooler_output
            summ = summ+ self.cos(q_e_output, a_output)
        summ = summ/entity_no
        summ = summ.unsqueeze(-1)
        label = label.unsqueeze(-1)
        concat = torch.cat((summ,label),dim=1)
        final = self.linear(concat)
      
        return final