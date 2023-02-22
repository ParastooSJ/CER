
from transformers import BertModel as bm
import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel


class CERCONTEXT(BertPreTrainedModel):
    def __init__(self,config):
        super(CERCONTEXT, self).__init__(config)
        self.bert_q = bm.from_pretrained('bert-base-uncased')
        self.bert_a = bm.from_pretrained('bert-base-uncased')
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.linear = nn.Linear(2, 1)
       
    def forward(self, input_ids, attention_mask,output_ids,output_attention_mask,label):
        q_output = self.bert_q(input_ids,attention_mask).pooler_output
        a_output = self.bert_a(output_ids,output_attention_mask).pooler_output
        
        join_output = self.cos(q_output, a_output)
        join_output = join_output.unsqueeze(-1)
        label = label.unsqueeze(-1)
        concat = torch.cat((join_output,label),dim=1)
        final = self.linear(concat)
        
        return final