import os
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from torchvision.models import resnet50
from torchvision import transforms
from argparse import ArgumentParser
from utils import *
from dataloders import test_dataloaders

def parse_option() :
    parser = ArgumentParser()
    
    # test arguments
    parser.add_argument('--test_shot', default=5, type=int, help='how many shots to test')
    parser.add_argument('--test_way', default=3, type=int, help='how many ways to test')
    parser.add_argument('--test_query', default=15, type=int, help='how many query to test')
    parser.add_argument('--dataset_nm', default='cifarfs', type=str)
    
    # models
    parser.add_argument('--model_path', default='./checkpoint/epoch50_loss1.414059302210808.pth', type=str)
    
    # gpu
    parser.add_argument('--n_gpu', default=2, type=int)
    opt = parser.parse_args()
    
    return opt


def main() :
    args = parse_option()
    
    # get pretrained model
    model = resnet50(pretrained = False)
    ckpt = torch.load(args.model_path)
    corrected_dict = {k.replace('module.', '') : v for k, v in ckpt.items() }
    model.load_state_dict(corrected_dict)
    model = model.cuda()

    print(f'>> check the trained model path : {args.model_path}')
    
    # get datasets
    test_dataset, test_dataset, test_tasks, test_loader = test_dataloaders(args)
    
    
    # start eval
    # check the test result
    loss_ctr = 0.0
    n_loss = 0.0
    n_acc = 0.0
    for i, batch in enumerate(test_loader, 1):
        loss, acc = fast_adapt(model, batch, args.test_way, args.test_shot, args.test_query, metric = pairwise_distances_logits, device = None)
        loss_ctr += 1
        n_acc += acc
        if i % 500 == 0 :
            print('batch {}: {:.2f}({:.2f})'.format(i, n_acc/loss_ctr * 100, acc * 100))

    print(f'Test Accuracy : {n_acc / loss_ctr * 100}')
    
    
if __name__ == "__main__" :
    main()
   