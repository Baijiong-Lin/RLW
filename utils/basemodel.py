import torch, sys
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, task_num, weighting=None, random_distribution=None):
        super(BaseModel, self).__init__()
        
        self.task_num = task_num
        self.weighting = weighting
        self.random_distribution = random_distribution
        
        self.rep_detach = False
            
        self.loss_weight_init = None
        
        if self.rep_detach:
            self.rep = [0]*self.task_num
            self.rep_i = [0]*self.task_num
        if isinstance(self.loss_weight_init, float):
            self.loss_scale = nn.Parameter(torch.FloatTensor([self.loss_weight_init]*self.task_num))
            
            
        if self.weighting == 'RLW' and self.random_distribution == 'random_normal':
            self.random_normal_mean, self.random_normal_std = torch.rand(self.task_num), torch.rand(self.task_num)
        else:
            self.random_normal_mean, self.random_normal_std = None, None
        
    def forward(self):
        pass
