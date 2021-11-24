import torch, sys
import torch.nn as nn
from transformers import BertModel
sys.path.append('../utils')
from basemodel import BaseModel

def compute_loss(logits, data, label_num, task_type=None, task=None):
    if task_type == 'TC' or task in ['panx', 'udpos']:
        loss_fct = nn.CrossEntropyLoss()
        if data['attention_mask'] is not None:
            active_loss = data['attention_mask'].view(-1) == 1
            active_logits = logits.view(-1, label_num)
            active_labels = torch.where(
                active_loss, data['labels'].view(-1), torch.tensor(loss_fct.ignore_index).type_as(data['labels'])
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, label_num), data['labels'].view(-1))
        return (loss, logits)
    elif task_type == 'SC' or task in ['xnli', 'pawsx']:
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, label_num), data['labels'].view(-1))
        return (loss, logits)
    else:
        raise('no support')

class mBert(BaseModel):
    def __init__(self, label_num, task_num, weighting, random_distribution, task_type='TC'):
        super(mBert, self).__init__(task_num=task_num,
                                   weighting=weighting,
                                   random_distribution=random_distribution)
        # task_type: TC(NER, POS), SC(XNIL, PAWSX)
        self.task_num = task_num
        self.label_num = label_num
        self.task_type = task_type
        
        add_pooling_layer = True if task_type == 'SC' else False
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased', add_pooling_layer=add_pooling_layer)
        
        self.dropout = nn.ModuleList([nn.Dropout(p=0.1, inplace=False) for _ in range(self.task_num)])
        self.fc = nn.ModuleList([nn.Linear(768, self.label_num) for _ in range(self.task_num)])
        
        
    def forward(self, data, task_index):
        outputs = self.bert(input_ids=data['input_ids'],
                           attention_mask=data['attention_mask'],
                           token_type_ids=data['token_type_ids'])
        rep = outputs[1] if self.task_type=='SC' else outputs[0]
        if self.rep_detach:
            self.rep[task_index] = rep
            self.rep_i[task_index] = rep.detach().clone()
            self.rep_i[task_index].requires_grad = True
            rep = self.rep_i[task_index]
        if self.task_type == 'TC':
            sequence_output = self.dropout[task_index](rep)
            logits = self.fc[task_index](sequence_output)
            loss = compute_loss(logits=logits,
                               task_type='TC',
                               data=data, label_num=self.label_num)
            return loss
            
        elif self.task_type == 'SC':
            pooled_output = self.dropout[task_index](rep)
            logits = self.fc[task_index](pooled_output)
            loss = compute_loss(logits=logits,
                               task_type='SC',
                               data=data, label_num=self.label_num)
            return loss
        else:
            raise('no support')
            

