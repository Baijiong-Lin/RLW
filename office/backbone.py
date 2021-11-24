import torch, sys
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../utils')
sys.path.append('../nyu')
from model import resnet
from basemodel import BaseModel


def build_model(model, weighting, random_distribution, task_num, class_num):
    if model == 'DMTL':
        model = DMTL(task_num=task_num, class_num=class_num,
                    weighting=weighting, random_distribution=random_distribution)
    return model

class DMTL(BaseModel):
    def __init__(self, task_num, base_net='resnet18', hidden_dim=512, class_num=31, 
                 random_distribution=False, weighting=None):
        super(DMTL, self).__init__(task_num=task_num,
                                   weighting=weighting,
                                   random_distribution=random_distribution)
        # base network
        self.base_network = resnet.__dict__[base_net](pretrained=True)
        # shared layer
        self.avgpool = self.base_network.avgpool
        self.hidden_layer_list = [nn.Linear(512, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.hidden_layer = nn.Sequential(*self.hidden_layer_list)
        # task-specific layer
        self.classifier_parameter = nn.Parameter(torch.FloatTensor(task_num, hidden_dim, class_num))

        # initialization
        self.hidden_layer[0].weight.data.normal_(0, 0.005)
        self.hidden_layer[0].bias.data.fill_(0.1)
        self.classifier_parameter.data.normal_(0, 0.01)

    def forward(self, inputs, task_index):
        features = self.base_network(inputs)
        features = torch.flatten(self.avgpool(features), 1)
        hidden_features = self.hidden_layer(features)
        rep = hidden_features
        if self.rep_detach:
            self.rep[task_index] = rep
            self.rep_i[task_index] = hidden_features.detach().clone()
            self.rep_i[task_index].requires_grad = True
            rep = self.rep_i[task_index]
        outputs = torch.mm(rep, self.classifier_parameter[task_index])
        return outputs

    def get_share_params(self):
        p = []
        p += self.base_network.parameters()
        p += self.hidden_layer.parameters()
        return p
