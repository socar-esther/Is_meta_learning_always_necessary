import os
import time
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from torchvision.models import resnet50
from torchvision import transforms

from argparse import ArgumentParser


# metric
def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits


# metric
def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

# train loop
def fast_adapt(model, batch, ways, shot, query_num, criterion, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    #if device is None:
        #device = model.cuda()
    data, labels = batch
    data = data.cuda()#.to(device)
    labels = labels.cuda() #.to(device)
    n_items = shot * ways

    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    loss = criterion(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc
