import torch, time, os, random, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from backbone import build_model

from data.pascal_context import PASCALContext
from data.custom_collate import collate_mil
from loss_functions import get_loss
from evaluation.evaluate_utils import PerformanceMeter, get_output

sys.path.append('../utils')
from weighting import weight_update

import argparse

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description= 'MTL for PASCAL')
    parser.add_argument('--data_root', default='', help='data root', type=str) 
    parser.add_argument('--gpu_id', default='0', help='gpu_id') 
    parser.add_argument('--random_distribution', default='normal', type=str, 
                        help='normal, random_normal, uniform, dirichlet, Bernoulli, Bernoulli_1')
    parser.add_argument('--weighting', default='EW', type=str, help='EW, RLW')
    return parser.parse_args()

params = parse_args()
print(params)

def adjust_learning_rate(optimizer, epoch, total_epoch=60):
    lr = 1e-4
    lambd = pow(1-(epoch/total_epoch), 0.9)
    lr = lr * lambd
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

tasks = ['semseg', 'human_parts', 'sal', 'normals']
total_epoch = 300

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

train_database = PASCALContext(root=params.data_root, split=['train'], aug=True,
                               do_edge='edge' in tasks,
                               do_human_parts='human_parts' in tasks,
                               do_semseg='semseg' in tasks,
                               do_normals='normals' in tasks,
                               do_sal='sal' in tasks)
test_database = PASCALContext(root=params.data_root, split=['val'], aug=False,
                              do_edge='edge' in tasks,
                              do_human_parts='human_parts' in tasks,
                              do_semseg='semseg' in tasks,
                              do_normals='normals' in tasks,
                              do_sal='sal' in tasks)

trainloader = DataLoader(train_database, batch_size=12, shuffle=True, drop_last=True,
                 num_workers=4, collate_fn=collate_mil)
testloader = DataLoader(test_database, batch_size=12, shuffle=False, drop_last=False,
                 num_workers=4)
        
model = build_model(tasks=tasks, dataset='PASCAL', model='DMTL', weighting=params.weighting,
                    random_distribution=params.random_distribution).cuda()
task_num = len(model.tasks)

criterion = {task: get_loss(task).cuda() for task in tasks}

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
train_batch = len(trainloader)
avg_cost = torch.zeros([total_epoch, 2*task_num])
lambda_weight = torch.ones([task_num, total_epoch, train_batch]).cuda()
for epoch in range(total_epoch):
    print('-'*10, epoch)
    s_t = time.time()
    
    adjust_learning_rate(optimizer, epoch, total_epoch)
    
    # iteration for all batches
    model.train()
    train_dataset = iter(trainloader)
    performance_meter = PerformanceMeter(tasks)
    for batch_index in range(train_batch):
        
        train_batch_data = train_dataset.next()
        train_data = train_batch_data['image'].cuda(non_blocking=True)
        targets = {task: train_batch_data[task].cuda(non_blocking=True) for task in tasks}
        
        train_pred = model(train_data)

        loss_train = torch.zeros(task_num).cuda()
        for tk, task in enumerate(tasks):
            loss_train[tk] = criterion[task](train_pred[task], targets[task])
            avg_cost[epoch, tk] += loss_train[tk].item()
            
        batch_weight = weight_update(params.weighting, loss_train, model, optimizer, epoch, 
                                     batch_index, task_num, clip_grad=False, scheduler=None, 
                                     random_distribution=params.random_distribution, 
                                     avg_cost=avg_cost[:,:task_num])
        if batch_weight is not None:
            lambda_weight[:, epoch, batch_index] = batch_weight
            
        performance_meter.update({t: get_output(train_pred[t], t) for t in tasks}, 
                                 {t: targets[t] for t in tasks})
    
    eval_results_train = performance_meter.get_score(verbose=False)   
    print('TRAIN:', eval_results_train)
    avg_cost[epoch, :task_num] /= train_batch
        

    # evaluating test data
    model.eval()
    with torch.no_grad():  # operations inside don't track history
        val_dataset = iter(testloader)
        val_batch = len(testloader)
        performance_meter = PerformanceMeter(tasks)
        for k in range(val_batch):
            val_batch_data = val_dataset.next()
            val_data = val_batch_data['image'].cuda(non_blocking=True)
            targets = {task: val_batch_data[task].cuda(non_blocking=True) for task in tasks}

            val_pred = model(val_data)
            for tk, task in enumerate(tasks):
                avg_cost[epoch, task_num+tk] += (criterion[task](val_pred[task], targets[task])).item()
            performance_meter.update({t: get_output(val_pred[t], t) for t in tasks}, 
                                 {t: targets[t] for t in tasks})
        eval_results_test = performance_meter.get_score(verbose=False)
        print('TEST:', eval_results_test)
        avg_cost[epoch, task_num:] /= val_batch

    e_t = time.time()
    print('TIME:', e_t-s_t)