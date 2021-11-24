import torch, random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def weight_update(weighting, loss_train, model, optimizer, epoch, batch_index, task_num,
                  clip_grad=False, scheduler=None, mgda_gn='l2', 
                  random_distribution=None, avg_cost=None):
    """
    weighting: weight method (EW, RLW)
    random_distribution: using in random (uniform, normal, random_normal, dirichlet, Bernoulli, Bernoulli_1)
    """
    batch_weight = None
    optimizer.zero_grad()
    if weighting == 'EW':
        batch_weight = torch.ones(task_num).cuda()
    elif weighting == 'RLW' and random_distribution is not None:
        if random_distribution == 'uniform':
            batch_weight = F.softmax(torch.rand(task_num).cuda(), dim=-1)
        elif random_distribution == 'normal':
            batch_weight = F.softmax(torch.randn(task_num).cuda(), dim=-1)
        elif random_distribution == 'dirichlet':
            # https://en.wikipedia.org/wiki/Dirichlet_distribution#Random_number_generation
            alpha = 1
            gamma_sample = [random.gammavariate(alpha, 1) for _ in range(task_num)]
            dirichlet_sample = [v / sum(gamma_sample) for v in gamma_sample]
            batch_weight = torch.Tensor(dirichlet_sample).cuda()
        elif random_distribution == 'random_normal':
            batch_weight = F.softmax(torch.normal(model.random_normal_mean, model.random_normal_std).cuda(), dim=-1)
        elif random_distribution == 'Bernoulli':
            while True:
                w = torch.randint(0, 2, (task_num,))
                if w.sum()!=0:
                    batch_weight = w.cuda()
                    break
        elif len(random_distribution.split('_'))==2 and random_distribution.split('_')[0]=='Bernoulli':
            w = random.sample(range(task_num), k=int(random_distribution.split('_')[1]))
            batch_weight = torch.zeros(task_num).cuda()
            batch_weight[w] = 1.
        else:
            raise('no support {}'.format(random_distribution))
    loss = torch.sum(loss_train*batch_weight)
#         optimizer.zero_grad()
    loss.backward()
    if clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    if weighting != 'EW' and batch_weight is not None and (batch_index+1) % 200 == 0:
        print('{} weight: {}'.format(weighting, batch_weight.cpu()))
    return batch_weight