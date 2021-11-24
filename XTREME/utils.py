import os
import numpy as np
import torch.nn as nn
from transformers import BertTokenizer
from transformers.data.processors.squad import SquadResult
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate
from seqeval.metrics import precision_score, recall_score, f1_score

from create_dataset import DataloaderSC, DataloaderTC
from processors.utils_tag import get_labels


def get_data(task, mode, all_dataloader, all_iter_dataloader):
    try:
        batch = all_iter_dataloader[task][mode].next()
    except:
        all_iter_dataloader[task][mode] = iter(all_dataloader[task][mode])
        batch = all_iter_dataloader[task][mode].next()
    batch = tuple(t.cuda(non_blocking=True) for t in batch if t is not None)
    inputs = {"input_ids": batch[0], 
              "attention_mask": batch[1], 
              "token_type_ids": batch[2]}
    inputs["labels"] = batch[3]
    return inputs


def get_metric(root_data, model, task, mode, all_dataloader, all_iter_dataloader, 
               squad_label=None, lg=None, lg_index=None):
    if lg is None:
        lg = task
    if lg_index is None:
        lg_index = task
    if task in ['panx', 'udpos']:
        for batch_index in range(len(all_dataloader[lg][mode])):
            inputs = get_data(lg, mode, all_dataloader, all_iter_dataloader)
            _, logits = model(inputs, lg_index)
            
            if batch_index == 0:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                
        preds = np.argmax(preds, axis=2)
        
        labels = get_labels('{}/{}/{}_processed_maxlen128/labels.txt'.format(root_data, task, task))
        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]
        
        pad_token_label_id = nn.CrossEntropyLoss().ignore_index
        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])
        precision = precision_score(out_label_list, preds_list)
        recall = recall_score(out_label_list, preds_list)
        f1 = f1_score(out_label_list, preds_list)
#         return precision, recall, f1
        return f1
    
    elif task in ['xnli', 'pawsx']:
        for batch_index in range(len(all_dataloader[lg][mode])):
            inputs = get_data(lg, mode, all_dataloader, all_iter_dataloader)
            _, logits = model(inputs, lg_index)
            
            if batch_index==0:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                
        #results[epoch, mode_index+1, lg_index, 0] /= (batch_index+1)
        preds = np.argmax(preds, axis=1)
        acc = (preds==out_label_ids).mean()
        return acc
    
    else:
        raise('no support!')
