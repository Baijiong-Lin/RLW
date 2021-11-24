import torch, time, os, random, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from backbone import build_model

from create_dataset import office_dataloader

sys.path.append('../utils')
from weighting import weight_update
import argparse
torch.set_num_threads(3)

torch.manual_seed(688)
random.seed(688)
np.random.seed(688)

def parse_args():
    parser = argparse.ArgumentParser(description= 'MTL for Office-31 and Office-Home')
    parser.add_argument('--data_root', default='', help='data root', type=str) 
    parser.add_argument('--dataset', default='office-31', type=str, help='office-31, office-home')
    parser.add_argument('--gpu_id', default='0', help='gpu_id') 
    parser.add_argument('--random_distribution', default='normal', type=str, 
        help='normal, random_normal, uniform, dirichlet, Bernoulli, Bernoulli_1')
    parser.add_argument('--weighting', default='EW', type=str, 
                        help='EW, RLW')
    return parser.parse_args()

params = parse_args()
print(params)

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

train_mode = 'trval'
if params.dataset == 'office-31':
    task_num, class_num = 3, 31
elif params.dataset == 'office-home':
    task_num, class_num = 4, 65
    
data_loader, iter_data_loader = office_dataloader(params.dataset, batchsize=64, root_path=params.data_root)

model = build_model('DMTL', params.weighting, params.random_distribution, task_num, class_num).cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  

total_epoch = 100
train_batch = max(len(data_loader[i][train_mode]) for i in range(task_num))
avg_cost = torch.zeros([total_epoch, task_num])
lambda_weight = torch.ones([task_num, total_epoch, train_batch]).cuda()
loss_fn = nn.CrossEntropyLoss().cuda()
best_val_acc, best_test_acc, early_count = 0, 0, 0
for epoch in range(total_epoch):
    print('--- Epoch {}'.format(epoch))
    s_t = time.time()
    model.train()
    for batch_index in range(train_batch):
        loss_train = torch.zeros(task_num).cuda()
        for task_index in range(task_num):
            try:
                train_data, train_label = iter_data_loader[task_index][train_mode].next()
            except:
                iter_data_loader[task_index][train_mode] = iter(data_loader[task_index][train_mode])
                train_data, train_label = iter_data_loader[task_index][train_mode].next()
            train_data, train_label = train_data.cuda(non_blocking=True), train_label.cuda(non_blocking=True)
            loss_train[task_index] = loss_fn(model(train_data, task_index), train_label)
            avg_cost[epoch, task_index] += loss_train[task_index].item()
            
        batch_weight = weight_update(params.weighting, loss_train, model, optimizer, epoch, 
                                     batch_index, task_num, clip_grad=False, scheduler=None, 
                                     random_distribution=params.random_distribution, 
                                     avg_cost=avg_cost)
        if batch_weight is not None:
            lambda_weight[:, epoch, batch_index] = batch_weight
            
    avg_cost[epoch] /= train_batch

    # evaluating test data
    model.eval()
    with torch.no_grad(): 
        right_num = np.zeros([2, task_num])
        count = np.zeros([2, task_num])
        loss_data_count = np.zeros([2, task_num])
        for mode_index, mode in enumerate(['val', 'test']):
            for k in range(task_num):
                for test_it, test_data in enumerate(data_loader[k][mode]):
                    x_test, y_test = test_data[0].cuda(non_blocking=True), test_data[1].cuda(non_blocking=True)
                    y_pred = model(x_test, k)
                    loss_t = loss_fn(y_pred, y_test)
                    loss_data_count[mode_index, k] += loss_t.item()
                    right_num[mode_index, k] += ((torch.max(F.softmax(y_pred, dim=-1), dim=-1)[1])==y_test).sum().item()
                    count[mode_index, k] += y_test.shape[0]
        acc_avg = (right_num/count).mean(axis=-1)
        loss_data_avg = (loss_data_count/count).mean(axis=-1)
        print('val acc {} {}, loss {}'.format(right_num[0]/count[0], acc_avg[0], loss_data_count[0]))
        print('test acc {} {}, loss {}'.format(right_num[1]/count[1], acc_avg[1], loss_data_count[1]))
    e_t = time.time()
    print('-- cost time {}'.format(e_t-s_t))
    if acc_avg[0] > best_val_acc:
        best_val_acc = acc_avg[0]
        early_count = 0
        print('-- -- epoch {} ; best val {} {} ; test acc {} {}'.format(epoch, right_num[0]/count[0], acc_avg[0],
                                                                         right_num[1]/count[1], acc_avg[1]))
    if acc_avg[1] > best_test_acc:
        best_test_acc = acc_avg[1]
        print('!! -- -- epoch {}; best test acc {} {}'.format(epoch, right_num[1]/count[1], acc_avg[1]))