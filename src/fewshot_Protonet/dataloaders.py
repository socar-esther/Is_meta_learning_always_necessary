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
from utils import pairwise_distances_logits, accuracy, fast_adapt

from argparse import ArgumentParser


# dataset
def get_datasets(args) :
    
    # dataset path 정의
    path_data = '../../datasets'
    train_dataset = l2l.vision.datasets.CIFARFS(root=path_data, mode='train',transform = transforms.ToTensor(), download = True)
    valid_dataset = l2l.vision.datasets.CIFARFS(root=path_data, mode='validation', transform = transforms.ToTensor(), download =True)
    test_dataset = l2l.vision.datasets.CIFARFS(root=path_data, mode='test', transform = transforms.ToTensor(), download = True)
    
    # get dataset and laoder
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
            NWays(train_dataset, args.train_way),
            KShots(train_dataset, args.train_query + args.train_shot), # train shot 적용
            LoadData(train_dataset),
            RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms, num_tasks = 20000)
    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)

    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
            NWays(valid_dataset, args.test_way),
            KShots(valid_dataset, args.test_query + args.test_shot),
            LoadData(valid_dataset),
            RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,task_transforms=valid_transforms,num_tasks=100) # 여기는 task 갯수 상관없음 (>200)
    valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)

    test_dataset = l2l.data.MetaDataset(test_dataset)
    test_transforms = [
            NWays(test_dataset, args.test_way),
            KShots(test_dataset, args.test_query + args.test_shot),  # test shot 적용
            LoadData(test_dataset),
            RemapLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,task_transforms=test_transforms,num_tasks=20000)
    test_loader = DataLoader(test_tasks, pin_memory=True, shuffle=True)
    
    return train_tasks, train_loader, valid_tasks, valid_loader, test_tasks, test_loader