import torch, time, os, random, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
from backbone import build_model
from utils import *

from create_dataset import CityScape

sys.path.append('../utils')
from weighting import weight_update

import argparse
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description= 'MTL for CityScapes')
    parser.add_argument('--data_root', default='', help='data root', type=str) 
    parser.add_argument('--gpu_id', default='0', help='gpu_id') 
    parser.add_argument('--model', default='DMTL', type=str, help='DMTL, MTAN')
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--random_distribution', default='normal', type=str, 
                        help='normal, uniform, random_random, dirichlet, Bernoulli, Bernoulli_1')
    parser.add_argument('--weighting', default='EW', type=str, help='EW, RLW')
    return parser.parse_args()

params = parse_args()
print(params)

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

if params.model == 'DMTL':
    batch_size = 64
    
cityscape_train_set = CityScape(root=params.data_root, mode='trainval', augmentation=params.aug)
cityscape_test_set = CityScape(root=params.data_root, mode='test', augmentation=False)

cityscape_test_loader = torch.utils.data.DataLoader(
    dataset=cityscape_test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True)


cityscape_train_loader = torch.utils.data.DataLoader(
    dataset=cityscape_train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True)

model = build_model(dataset='CityScape', model=params.model, 
                    weighting=params.weighting, random_distribution=params.random_distribution).cuda()
task_num = len(model.tasks)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR')
total_epoch = 100
train_batch = len(cityscape_train_loader)
avg_cost = torch.zeros([total_epoch, 12])
lambda_weight = torch.ones([task_num, total_epoch, train_batch]).cuda()
for epoch in range(total_epoch):
    s_t = time.time()
    cost = torch.zeros(12)

    # iteration for all batches
    model.train()
    train_dataset = iter(cityscape_train_loader)
    conf_mat = ConfMatrix(model.class_nb)
    for batch_index in range(train_batch):
        train_data, train_label, train_depth = train_dataset.next()
        train_data, train_label = train_data.cuda(non_blocking=True), train_label.long().cuda(non_blocking=True)
        train_depth = train_depth.cuda(non_blocking=True)

        train_pred = model(train_data)

        train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                      model_fit(train_pred[1], train_depth, 'depth')]
        loss_train = torch.zeros(task_num).cuda()
        for i in range(task_num):
            loss_train[i] = train_loss[i]
        
        batch_weight = weight_update(params.weighting, loss_train, model, optimizer, epoch, 
                                     batch_index, task_num, clip_grad=False, scheduler=None, 
                                     random_distribution=params.random_distribution, 
                                     avg_cost=avg_cost[:,0:4:3])
        if batch_weight is not None:
            lambda_weight[:, epoch, batch_index] = batch_weight

        # accumulate label prediction for every pixel in training images
        conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

        cost[0] = train_loss[0].item()
        cost[3] = train_loss[1].item()
        cost[4], cost[5] = depth_error(train_pred[1], train_depth)
        avg_cost[epoch, :6] += cost[:6] / train_batch

    # compute mIoU and acc
    avg_cost[epoch, 1], avg_cost[epoch, 2] = conf_mat.get_metrics()

    # evaluating test data
    model.eval()
    conf_mat = ConfMatrix(model.class_nb)
    with torch.no_grad():  # operations inside don't track history
        val_dataset = iter(cityscape_test_loader)
        val_batch = len(cityscape_test_loader)
        for k in range(val_batch):
            val_data, val_label, val_depth = val_dataset.next()
            val_data, val_label = val_data.cuda(non_blocking=True), val_label.long().cuda(non_blocking=True)
            val_depth = val_depth.cuda(non_blocking=True)

            val_pred = model(val_data)
            val_loss = [model_fit(val_pred[0], val_label, 'semantic'),
                         model_fit(val_pred[1], val_depth, 'depth')]

            conf_mat.update(val_pred[0].argmax(1).flatten(), val_label.flatten())

            cost[6] = val_loss[0].item()
            cost[9] = val_loss[1].item()
            cost[10], cost[11] = depth_error(val_pred[1], val_depth)
            avg_cost[epoch, 6:] += cost[6:] / val_batch

        # compute mIoU and acc
        avg_cost[epoch, 7], avg_cost[epoch, 8] = conf_mat.get_metrics()
    
    scheduler.step()
    e_t = time.time()
    print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} ||'
        'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} || {:.4f}'
        .format(epoch, avg_cost[epoch, 0], avg_cost[epoch, 1], avg_cost[epoch, 2], avg_cost[epoch, 3],
                avg_cost[epoch, 4], avg_cost[epoch, 5], avg_cost[epoch, 6], avg_cost[epoch, 7], avg_cost[epoch, 8],
                avg_cost[epoch, 9], avg_cost[epoch, 10], avg_cost[epoch, 11], e_t-s_t))