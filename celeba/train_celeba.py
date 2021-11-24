import torch, os, random, copy, time, sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score,f1_score
from celeba_loader import CELEBA
from multi_faces_resnet import build_model

sys.path.append('../utils')
from weighting import weight_update

import argparse

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description= 'MTL for CelebA')
    parser.add_argument('--data_root', default='', help='data root', type=str) 
    parser.add_argument('--gpu_id', default='0', help='gpu_id') 
    parser.add_argument('--random_distribution', default='normal', type=str, 
                        help='normal, uniform, random_normal, dirichlet, Bernoulli, Bernoulli_1')
    parser.add_argument('--weighting', default='EW', type=str, help='EW, RLW')
    return parser.parse_args()

params = parse_args()
print(params)
os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

task_num = 40
batch_size = 512
configs = {'path':params.data_root,
          'img_rows':64,
          'img_cols':64,
          'batch_size':batch_size}
train_dst = CELEBA(root=configs['path'], is_transform=True, split='train', img_size=(configs['img_rows'], configs['img_cols']), augmentations=None)
val_dst = CELEBA(root=configs['path'], is_transform=True, split='val', img_size=(configs['img_rows'], configs['img_cols']), augmentations=None)
test_dst = CELEBA(root=configs['path'], is_transform=True, split='test', img_size=(configs['img_rows'], configs['img_cols']), augmentations=None)

train_loader = torch.utils.data.DataLoader(train_dst, batch_size=configs['batch_size'], shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dst, batch_size=configs['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dst, batch_size=configs['batch_size'], num_workers=2, pin_memory=True)
   

model = build_model(task_num, 'DMTL', params.weighting, params.random_distribution).cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_fn = nn.CrossEntropyLoss(reduction='none').cuda()

total_epoch = 150
avg_cost = torch.zeros([total_epoch, task_num])
lambda_weight = torch.ones([task_num, total_epoch, len(train_loader)]).cuda()
best_test_acc, best_test_epoch, best_val_acc, best_val_epoch, best_val_test_acc = 0, 0, 0, 0, 0
for epoch in range(total_epoch):
    s_t = time.time()
    print('Epoch {}'.format(epoch), '-'*20)
    for batch_idx, train_batch in enumerate(train_loader):
#         if batch_idx==10:
#             break

        model.train()
    
        y_pred = model(train_batch[0].cuda(non_blocking=True)).reshape(-1, 2)
        y_true = torch.cat(train_batch[1:1+task_num]).long().cuda(non_blocking=True)
        loss_train = loss_fn(y_pred, y_true).reshape(task_num, -1).mean(-1)
        for i in range(task_num):
            avg_cost[epoch, i] += loss_train[i].item()*train_batch[1].size()[0]
            
        batch_weight = weight_update(params.weighting, loss_train, model, optimizer, epoch, 
                                     batch_idx, task_num, clip_grad=False, scheduler=None, 
                                     random_distribution=params.random_distribution, 
                                     avg_cost=avg_cost)
        if batch_weight is not None:
            lambda_weight[:, epoch, batch_idx] = batch_weight
        
        #####
                
        if (batch_idx+1)%20 == 0:
            print('\t Train loss {:.4f} | {}/{}'.format((avg_cost[epoch]/batch_idx).mean(), batch_idx, len(train_loader)))
            
    avg_cost[epoch] /= len(train_loader)
    
    # evaluation 
    with torch.no_grad():
        model.eval()
        acc_avg, f1_avg = np.zeros([2, task_num]), np.zeros([2, task_num])
        for idx, (mode, dataloader) in enumerate(zip(['val', 'test'], [val_loader, test_loader])):
            for batch_idx, test_batch in enumerate(dataloader):
                model_pred = model(test_batch[0].cuda(non_blocking=True))
                if batch_idx==0:
                    y_true = torch.cat(test_batch[1:1+task_num]).long().reshape(task_num, -1).numpy()
                    y_pred = (torch.max(F.softmax(model_pred, dim=-1), dim=-1)[1]).cpu().numpy()
                else:
                    y_true_ = torch.cat(test_batch[1:1+task_num]).long().reshape(task_num, -1).numpy()
                    y_true = np.concatenate([y_true, y_true_], axis=-1)
                    y_pred_ = (torch.max(F.softmax(model_pred, dim=-1), dim=-1)[1]).cpu().numpy()
                    y_pred = np.concatenate([y_pred, y_pred_], axis=-1)
            for tn in range(task_num):
                acc_avg[idx][tn] = accuracy_score(y_true[tn], y_pred[tn])
                f1_avg[idx][tn] = f1_score(y_true[tn], y_pred[tn])
        acc_avg, f1_avg = acc_avg.mean(-1), f1_avg.mean(-1)
        print('\t Val Avg Acc {:.4f} Avg F1 {:.4f} | Test Avg Acc {:.4f} Avg F1 {:.4f}'.format(acc_avg[0], 
                                                                               f1_avg[0], 
                                                                               acc_avg[1], f1_avg[1]))
    e_t = time.time()
    print('-- cost time {}'.format(e_t-s_t))
    if acc_avg[0] > best_val_acc:
        best_val_acc = acc_avg[0]
        best_val_test_acc = acc_avg[1]
        best_val_epoch = epoch
    if acc_avg[1] > best_test_acc:
        best_test_acc = acc_avg[1]
        best_test_epoch = epoch
    print('++ -- Best val acc {:.4f} | Test acc {:.4f} | Epoch {}'.format(best_val_acc, best_val_test_acc, best_val_epoch))
    print('!! -- Best Test acc {:.4f} | Epoch {}'.format(best_test_acc, best_test_epoch))
            