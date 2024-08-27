# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:53:25 2024

@author: 86150
"""

import torch
from torch import nn

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')

torch.cuda.device_count()

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()

x = torch.tensor([1, 2, 3])
x.device

X = torch.ones(2, 3, device=try_gpu())
X


Z = X.cuda(0)
print(X)
print(Z)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())

