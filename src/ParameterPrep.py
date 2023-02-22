
import torch
import torch.nn as nn
from pytorch_pretrained_bert.optimization import BertAdam
from tqdm.auto import tqdm
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger("Qblink-20-bert-regressor")


class FreezableBertAdam(BertAdam):
    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    continue
                lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr
        
        
def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))

def count_model_parameters(model):
    logger.info(
        "# of paramters: {:,d}".format(
            sum(p.numel() for p in model.parameters())))
    logger.info(
        "# of trainable paramters: {:,d}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)))


class PrepareOptimizer:
    def __init__(self,model):
        self.param_optimizer = list(model.named_parameters())
        self.no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in self.param_optimizer if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.param_optimizer if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]

    def get_optimizer(self,num_train_optimization_steps: int, learning_rate: float):
        WARMUP_PROPORTION = 0.1
        grouped_parameters = [
        x for x in self.optimizer_grouped_parameters if any([p.requires_grad for p in x["params"]])
        ]
        for group in grouped_parameters:
            group['lr'] = learning_rate
        
        optimizer = FreezableBertAdam(grouped_parameters,
                            lr=learning_rate, warmup=WARMUP_PROPORTION,
                            t_total=num_train_optimization_steps)
        return optimizer


