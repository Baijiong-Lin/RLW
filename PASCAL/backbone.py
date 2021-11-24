import torch, sys
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../nyu')
from model import resnet
from model.resnet_dilated import ResnetDilated
from model.aspp import DeepLabHead
from model.resnet import Bottleneck, conv1x1

sys.path.append('../utils')
from basemodel import BaseModel

def build_model(tasks, dataset, model, weighting, random_distribution):
    if model == 'DMTL':
        model = DeepLabv3(tasks, dataset=dataset, weighting=weighting, random_distribution=random_distribution)
    return model

class DeepLabv3(BaseModel):
    def __init__(self, tasks, dataset='PASCAL', weighting=False, random_distribution=None):

        if dataset == 'PASCAL':
            self.class_nb = 21
            self.tasks = tasks
            self.num_out_channels = {'semseg': 21, 'human_parts': 7, 'sal': 1,
                                     'normals': 3, 'edge': 1}
        else:
            raise('No support {} dataset'.format(dataset))
        self.task_num = len(self.tasks)
        
        super(DeepLabv3, self).__init__(task_num=self.task_num,
                                        weighting=weighting, 
                                        random_distribution=random_distribution)
        
        self.backbone = ResnetDilated(resnet.__dict__['resnet18'](pretrained=True))
        self.decoders = nn.ModuleList([DeepLabHead(512, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x):
        img_size  = x.size()[-2:]
        x = self.backbone(x)
        self.rep = x
        if self.rep_detach:
            for tn in range(self.task_num):
                self.rep_i[tn] = self.rep.detach().clone()
                self.rep_i[tn].requires_grad = True
        out = {}
        for i, t in enumerate(self.tasks):
            out[t] = F.interpolate(self.decoders[i](self.rep_i[i] if self.rep_detach else x), 
                                   img_size, mode='bilinear', align_corners=True)
        return out
    
    def get_share_params(self):
        return self.backbone.parameters()
