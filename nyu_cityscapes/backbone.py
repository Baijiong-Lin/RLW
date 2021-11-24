import torch, sys
import torch.nn as nn
import torch.nn.functional as F

from model import resnet
from model.resnet_dilated import ResnetDilated
from model.aspp import DeepLabHead
from model.resnet import Bottleneck, conv1x1

sys.path.append('../utils')
from basemodel import BaseModel

def build_model(dataset, model, weighting, random_distribution=None):
    if model == 'DMTL':
        model = DeepLabv3(dataset=dataset, weighting=weighting, random_distribution=random_distribution)
    elif model == 'MTAN':
        model = MTANDeepLabv3(dataset=dataset, weighting=weighting, random_distribution=random_distribution)
    elif model == 'NDDRCNN':
        model = NDDRCNN(dataset=dataset, weighting=weighting, random_distribution=random_distribution)
    elif model == 'Cross_Stitch':
        model = Cross_Stitch(dataset=dataset, weighting=weighting, random_distribution=random_distribution)
    return model

class DeepLabv3(BaseModel):
    def __init__(self, dataset='NYUv2', weighting=None, random_distribution=None):
        
        ch = [256, 512, 1024, 2048]
        
        if dataset == 'NYUv2':
            self.class_nb = 13
            self.tasks = ['segmentation', 'depth', 'normal']
            self.num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3}
        elif dataset == 'CityScape':
            self.class_nb = 7
            self.tasks = ['segmentation', 'depth']
            self.num_out_channels = {'segmentation': 7, 'depth': 1}
        else:
            raise('No support {} dataset'.format(dataset))
        self.task_num = len(self.tasks)
        
        super(DeepLabv3, self).__init__(task_num=self.task_num,
                                        weighting=weighting,
                                        random_distribution=random_distribution)
        
        self.backbone = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
        self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x):
        img_size  = x.size()[-2:]
        x = self.backbone(x)
        self.rep = x
        if self.rep_detach:
            for tn in range(self.task_num):
                self.rep_i[tn] = self.rep.detach().clone()
                self.rep_i[tn].requires_grad = True
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](self.rep_i[i] if self.rep_detach else x), 
                                   img_size, mode='bilinear', align_corners=True)
            if t in ['segmentation', 'segment_semantic']:
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
    
    def get_share_params(self):
        return self.backbone.parameters()

class MTANDeepLabv3(BaseModel):
    def __init__(self, dataset='NYUv2', weighting=None, random_distribution=None):
        
        ch = [256, 512, 1024, 2048]
        
        if dataset == 'NYUv2':
            self.class_nb = 13
            self.tasks = ['segmentation', 'depth', 'normal']
            self.num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3}
        elif dataset == 'CityScape':
            self.class_nb = 7
            self.tasks = ['segmentation', 'depth']
            self.num_out_channels = {'segmentation': 7, 'depth': 1}
        else:
            raise('No support {} dataset'.format(dataset))
        self.task_num = len(self.tasks)
        
        super(MTANDeepLabv3, self).__init__(task_num=self.task_num,
                                        weighting=weighting, 
                                        random_distribution=random_distribution)
        
        backbone = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)

        # We will apply the attention over the last bottleneck layer in the ResNet. 
        self.shared_layer1_b = backbone.layer1[:-1] 
        self.shared_layer1_t = backbone.layer1[-1]

        self.shared_layer2_b = backbone.layer2[:-1]
        self.shared_layer2_t = backbone.layer2[-1]

        self.shared_layer3_b = backbone.layer3[:-1]
        self.shared_layer3_t = backbone.layer3[-1]

        self.shared_layer4_b = backbone.layer4[:-1]
        self.shared_layer4_t = backbone.layer4[-1]

        # Define task specific attention modules using a similar bottleneck design in residual block
        # (to avoid large computations)
        self.encoder_att_1 = nn.ModuleList([self.att_layer(ch[0], ch[0] // 4, ch[0]) for _ in self.tasks])
        self.encoder_att_2 = nn.ModuleList([self.att_layer(2 * ch[1], ch[1] // 4, ch[1]) for _ in self.tasks])
        self.encoder_att_3 = nn.ModuleList([self.att_layer(2 * ch[2], ch[2] // 4, ch[2]) for _ in self.tasks])
        self.encoder_att_4 = nn.ModuleList([self.att_layer(2 * ch[3], ch[3] // 4, ch[3]) for _ in self.tasks])

        # Define task shared attention encoders using residual bottleneck layers
        # We do not apply shared attention encoders at the last layer,
        # so the attended features will be directly fed into the task-specific decoders.
        self.encoder_block_att_1 = self.conv_layer(ch[0], ch[1] // 4)
        self.encoder_block_att_2 = self.conv_layer(ch[1], ch[2] // 4)
        self.encoder_block_att_3 = self.conv_layer(ch[2], ch[3] // 4)
        
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
        
    def forward(self, x, task_index=None):
        img_size  = x.size()[-2:]
        # Shared convolution
        x = self.shared_conv(x)
        
        # Shared ResNet block 1
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)

        # Shared ResNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)

        # Shared ResNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)
        
        # Shared ResNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)

        # Attention block 1 -> Apply attention over last residual block
        a_1_mask = [att_i(u_1_b) for att_i in self.encoder_att_1]  # Generate task specific attention map
        a_1 = [a_1_mask_i * u_1_t for a_1_mask_i in a_1_mask]  # Apply task specific attention map to shared features
        a_1 = [self.down_sampling(self.encoder_block_att_1(a_1_i)) for a_1_i in a_1]
        
        # Attention block 2 -> Apply attention over last residual block
        a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)]
        a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [self.encoder_block_att_2(a_2_i) for a_2_i in a_2]
        
        # Attention block 3 -> Apply attention over last residual block
        a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
        a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [self.encoder_block_att_3(a_3_i) for a_3_i in a_3]
        
        # Attention block 4 -> Apply attention over last residual block (without final encoder)
        a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_att_4)]
        a_4 = [a_4_mask_i * u_4_t for a_4_mask_i in a_4_mask]
        
        rep = a_4
        if self.rep_detach:
            for tn in range(self.task_num):
                self.rep[tn] = a_4[tn]
                self.rep_i[tn] = a_4[tn].detach().clone()
                self.rep_i[tn].requires_grad = True
                rep[tn] = self.rep_i[tn]
        
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](rep[i]), size=img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
    
    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid())
        
    def conv_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(conv1x1(in_channel, 4 * out_channel, stride=1),
                                   nn.BatchNorm2d(4 * out_channel))
        return Bottleneck(in_channel, out_channel, downsample=downsample)
        
    
    def get_share_params(self):
        p = []
        p += self.shared_conv.parameters()
        p += self.shared_layer1_b.parameters()
        p += self.shared_layer2_b.parameters()
        p += self.shared_layer3_b.parameters()
        p += self.shared_layer4_b.parameters()
        p += self.shared_layer1_t.parameters()
        p += self.shared_layer2_t.parameters()
        p += self.shared_layer3_t.parameters()
        p += self.shared_layer4_t.parameters()
        p += self.encoder_att_1.parameters()
        p += self.encoder_att_2.parameters()
        p += self.encoder_att_3.parameters()
        p += self.encoder_att_4.parameters()
        p += self.encoder_block_att_1.parameters()
        p += self.encoder_block_att_2.parameters()
        p += self.encoder_block_att_3.parameters()
        p += self.down_sampling.parameters()
        return p
    
class NDDRLayer(nn.Module):
    def __init__(self, tasks, channels, alpha, beta):
        super(NDDRLayer, self).__init__()
        self.tasks = tasks
        self.layer = nn.ModuleDict({task: nn.Sequential(
                                        nn.Conv2d(len(tasks) * channels, channels, 1, 1, 0, bias=False), nn.BatchNorm2d(channels, momentum=0.05), nn.ReLU()) for task in self.tasks}) # Momentum set as NDDR-CNN repo
        
        # Initialize
        for i, task in enumerate(self.tasks):
            layer = self.layer[task]
            t_alpha = torch.diag(torch.FloatTensor([alpha for _ in range(channels)])) # C x C
            t_beta = torch.diag(torch.FloatTensor([beta for _ in range(channels)])).repeat(1, len(self.tasks)) # C x (C x T)
            t_alpha = t_alpha.view(channels, channels, 1, 1)
            t_beta = t_beta.view(channels, channels * len(self.tasks), 1, 1)
    
            layer[0].weight.data.copy_(t_beta)
            layer[0].weight.data[:,int(i*channels):int((i+1)*channels)].copy_(t_alpha)
            layer[1].weight.data.fill_(1.0)
            layer[1].bias.data.fill_(0.0)


    def forward(self, x):
        x = torch.cat([x[task] for task in self.tasks], 1) # Use self.tasks to retain order!
        output = {task: self.layer[task](x) for task in self.tasks}
        return output


class NDDRCNN(BaseModel):
    def __init__(self, dataset='NYUv2', weighting=None, random_distribution=None):
#         super(NDDRCNN, self).__init__()

        # ch = [256, 512, 1024, 2048]

        if dataset == 'NYUv2':
            self.class_nb = 13
            self.tasks = ['segmentation', 'depth', 'normal']
            self.num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3}
        elif dataset == 'CityScape':
            self.class_nb = 7
            self.tasks = ['segmentation', 'depth']
            self.num_out_channels = {'segmentation': 7, 'depth': 1}
        else:
            raise('No support {} dataset'.format(dataset))
            
        self.task_num = len(self.tasks)
        
        super(NDDRCNN, self).__init__(task_num=self.task_num,
                                        weighting=weighting, 
                                        random_distribution=random_distribution)
        
        self.backbone = nn.ModuleDict({t: ResnetDilated(resnet.__dict__['resnet50'](pretrained=True)) for t in self.tasks})
        self.heads = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])

        self.all_stages = ['conv', 'layer1_without_conv', 'layer2', 'layer3', 'layer4']
        self.nddr_stages = ['conv', 'layer1_without_conv', 'layer2', 'layer3', 'layer4']
        channels = {'conv': 64, 'layer1_without_conv': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}

        alpha = 0.9
        beta = 0.1

        # NDDR-CNN units
        self.nddr = nn.ModuleDict({stage: NDDRLayer(self.tasks, channels[stage], alpha, beta) for stage in self.nddr_stages})


    def forward(self, x):
        img_size = x.size()[-2:]
        x = {task: x for task in self.tasks} # Feed as input to every single-task network

        # Backbone
        for stage in self.all_stages:
    
            # Forward through next stage of task-specific network
            for task in self.tasks:
                x[task] = self.backbone[task].forward_stage(x[task], stage)
            
            if stage in self.nddr_stages:
                # Fuse task-specific features through NDDR-layer.
                x = self.nddr[stage](x)
        
        rep = x
        if self.rep_detach:
            for tn, t in enumerate(self.tasks):
                self.rep[tn] = x[t]
                self.rep_i[tn] = x[t].detach().clone()
                self.rep_i[tn].requires_grad = True
                rep[t] = self.rep_i[tn]
                
        # Task-specific heads
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.heads[i](rep[t]), img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)

        return out

    def get_share_params(self):
        p = []
        p += self.backbone.parameters()
        p += self.nddr.parameters()
        return p
    
    
class Cross_Stitch(BaseModel):
    def __init__(self, dataset='NYUv2', weighting=None, random_distribution=None):
#         super(Cross_Stitch, self).__init__()
        
        if dataset == 'NYUv2':
            self.class_nb = 13
            self.tasks = ['segmentation', 'depth', 'normal']
            self.num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3}
        elif dataset == 'CityScape':
            self.class_nb = 7
            self.tasks = ['segmentation', 'depth']
            self.num_out_channels = {'segmentation': 7, 'depth': 1}
        else:
            raise('No support {} dataset'.format(dataset))
            
        self.task_num = len(self.tasks)
        
        super(Cross_Stitch, self).__init__(task_num=self.task_num,
                                        weighting=weighting, 
                                        random_distribution=random_distribution)
        
        backbone = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)
        
        backbones = nn.ModuleList([ResnetDilated(resnet.__dict__['resnet50'](pretrained=True)) for _ in self.tasks])
        ch = [256, 512, 1024, 2048]

        # We will apply the cross-stitch unit over the last bottleneck layer in the ResNet. 
        self.resnet_layer1 = nn.ModuleList([])
        self.resnet_layer2 = nn.ModuleList([])
        self.resnet_layer3 = nn.ModuleList([])
        self.resnet_layer4 = nn.ModuleList([])
        for i in range(len(self.tasks)):
            self.resnet_layer1.append(backbones[i].layer1) 
            self.resnet_layer2.append(backbones[i].layer2)
            self.resnet_layer3.append(backbones[i].layer3)
            self.resnet_layer4.append(backbones[i].layer4)
        del backbone, backbones
        # define cross-stitch units
        self.cross_unit = nn.Parameter(torch.ones(4, 3))
        
#         self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x):
        img_size  = x.size()[-2:]
        # Shared convolution
        x = self.shared_conv(x)
        
        # ResNet blocks with cross-stitch
        res_feature = [0, 0, 0]
        for j in range(3):
            res_feature[j] = [0, 0, 0, 0]
               
        for i in range(4):
            if i == 0:
                res_layer = self.resnet_layer1
            elif i == 1:
                res_layer = self.resnet_layer2
            elif i == 2:
                res_layer = self.resnet_layer3
            elif i == 3:
                res_layer = self.resnet_layer4
            for j in range(3):
                if i == 0:
                    res_feature[j][i] = res_layer[j](x)
                else:
                    cross_stitch = self.cross_unit[i - 1][0] * res_feature[0][i - 1] + \
                                   self.cross_unit[i - 1][1] * res_feature[1][i - 1] + \
                                   self.cross_unit[i - 1][2] * res_feature[2][i - 1]
                    res_feature[j][i] = res_layer[j](cross_stitch)
                    
        rep = res_feature
        if self.rep_detach:
            for tn, t in enumerate(self.tasks):
                self.rep[tn] = res_feature[tn][-1]
                self.rep_i[tn] = res_feature[tn][-1].detach().clone()
                self.rep_i[tn].requires_grad = True
            
        # Task specific decoders
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](rep[i][-1]), size=img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
        
    def get_share_params(self):
        p = []
        p += self.shared_conv.parameters()
        p += self.resnet_layer1.parameters()
        p += self.resnet_layer2.parameters()
        p += self.resnet_layer3.parameters()
        p += self.resnet_layer4.parameters()
        p.append(self.cross_unit)
        return p
