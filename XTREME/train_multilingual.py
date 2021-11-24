import random, os, time, argparse, sys, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup, logging
logging.set_verbosity_error()
logging.set_verbosity_warning()
from tqdm import tqdm

from create_dataset import DataloaderSC, DataloaderTC
from model import mBert
from utils import get_data, get_metric
sys.path.append('../utils')
from weighting import weight_update

torch.manual_seed(688)
random.seed(688)
np.random.seed(688)

def parse_args():
    parser = argparse.ArgumentParser(description= 'mBert for multilingual tasks')
    parser.add_argument('--dataset', default='xnli', type=str, help='xnli, pawsx, panx, udpos')
    parser.add_argument('--gpu_id', default='0', help='gpu_id') 
    parser.add_argument('--random_distribution', default='normal', type=str, 
                        help='normal, uniform, random_random, dirichlet, Bernoulli, Bernoulli_1')
    parser.add_argument('--weighting', default='EW', type=str, help='EW, RLW')
    return parser.parse_args()

params = parse_args()
print(params)

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

if params.dataset == 'udpos':
    lang_list = ['en', 'zh', 'te', 'vi']
else:
    lang_list = ['en', 'zh', 'de', 'es']
task_num = len(lang_list)
model_name_or_path = 'bert-base-multilingual-cased'
model_type = 'bert'
mode_list = ['train', 'dev', 'test']
max_seq_length = 128

root_data = './data/'

batch_size = 32
if params.dataset in ['xnli', 'pawsx']:
    task_type = 'SC'
    data_dir = '{}/{}'.format(root_data, params.dataset)
    dataloader, iter_dataloader, labels = DataloaderSC(lang_list=lang_list,
                                                  model_name_or_path=model_name_or_path,
                                                  model_type=model_type,
                                                  mode_list=mode_list,
                                                  data_dir=data_dir,
                                                  max_seq_length=max_seq_length,
                                                  batch_size=batch_size)
elif params.dataset in ['panx', 'udpos']:
    task_type = 'TC'
    data_dir = '{}/{}/{}_processed_maxlen128/'.format(root_data, params.dataset, params.dataset)
    dataloader, iter_dataloader, labels = DataloaderTC(lang_list=lang_list,
                                                  model_name_or_path=model_name_or_path,
                                                  model_type=model_type,
                                                  mode_list=mode_list,
                                                  data_dir=data_dir,
                                                  max_seq_length=max_seq_length,
                                                  batch_size=batch_size)
else:
    raise('no support！')


model = mBert(label_num=len(labels), 
              task_num=task_num, 
              task_type=task_type, 
              weighting=params.weighting,
              random_distribution=params.random_distribution).cuda()

total_epoch = 500
train_batch = max(len(dataloader[lg]['train']) for lg in lang_list)
t_total = train_batch*total_epoch

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0, 
                                            num_training_steps=t_total)

best_dev_acc, best_dev_epoch, early_count = 0, 0, 0
lambda_weight = torch.ones([task_num, total_epoch, train_batch]).cuda()
results = np.zeros([total_epoch, 3, task_num])
for epoch in range(total_epoch):
    print('--- Epoch {}'.format(epoch))
    s_t = time.time()
    model.train()
    for batch_index in range(train_batch):
#         if batch_index > 2:
#             break
        loss_train = torch.zeros(task_num).cuda()
        for lg_index, lg in enumerate(lang_list):
            inputs = get_data(lg, 'train', dataloader, iter_dataloader)
            outputs = model(inputs, lg_index)
            loss_train[lg_index] = outputs[0]
            results[epoch, 0, lg_index] += outputs[0].item()
                
        batch_weight = weight_update(params.weighting, loss_train, model, optimizer, epoch, 
                                     batch_index, task_num, clip_grad=True, scheduler=scheduler, 
                                     random_distribution=params.random_distribution, 
                                     avg_cost=results[:,0,:])
        if batch_weight is not None:
            lambda_weight[:, epoch, batch_index] = batch_weight

    results[epoch, 0, :] /= (batch_index+1)
    print('Train Loss {}'.format(results[epoch,0,:].mean()))
        
    model.eval()
    with torch.no_grad():
        for mode_index, mode in enumerate(['dev', 'test']):
            for lg_index, lg in enumerate(lang_list):
                results[epoch, mode_index+1, lg_index] = get_metric(root_data, model, params.dataset, mode, 
                                                           dataloader, iter_dataloader, lg=lg, lg_index=lg_index)
                
    e_t = time.time()
    print('Dev Acc/F1 {} avg {}'.format(results[epoch,1,:], 
                                    results[epoch,1,:].mean()))
    print('Test Acc/F1 {} avg {}'.format(results[epoch,2,:], 
                                     results[epoch,2,:].mean()))
    print('cost time {}'.format(e_t-s_t))
    
    if results[epoch,1,:].mean() > best_dev_acc:
        best_dev_acc = results[epoch,1,:].mean()
        best_dev_epoch = epoch
    print('Best Dev Epoch {}'.format(best_dev_epoch))