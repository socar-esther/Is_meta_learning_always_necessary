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
from dataloaders import get_datasets

from argparse import ArgumentParser

def parse_option() :
    parser = ArgumentParser()
    
    # train arguments
    parser.add_argument('--max_epoch', default=200, type=int, help='how many epochs to train')
    parser.add_argument('--train_shot', default=50, type=int, help='how many shots to train')
    parser.add_argument('--train_way', default=3, type=int, help='how many ways to train')
    parser.add_argument('--train_query', default=15, type=int, help='how many query to train')
    
    # test arguments
    parser.add_argument('--test_shot', default=5, type=int, help='how many shots to test')
    parser.add_argument('--test_way', default=3, type=int, help='how many ways to test')
    parser.add_argument('--test_query', default=15, type=int, help='how many query to test')
    
    # gpu
    parser.add_argument('--n_gpu', default=2, type=int)
    opt = parser.parse_args()
    
    return opt


def main() :    
    args = parse_option()
    
    # model setting (not Imagenet pretrained)
    model = resnet50(pretrained = False)
    model.cuda()
    
    # multi gpus check
    if torch.cuda.is_available():
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True
    
    # dataset settings
    train_tasks, train_loader, valid_tasks, valid_loader, test_tasks, test_loader = get_datasets(args)
    
    # optimizer settings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    
    # 학습시작
    best_loss = 10000000.0
    print('Start Training Phase !! ')
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        loss_ctr = 0
        n_loss = 0
        n_acc = 0

        for i in range(100):
            batch = next(iter(train_loader))
            loss, acc = fast_adapt(model, batch, args.train_way, args.train_shot, args.train_query, criterion, metric = pairwise_distances_logits, device = None )
            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        print('###################')
        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(epoch, n_loss/loss_ctr, n_acc/loss_ctr))

        ## start validation phase
        model.eval()
        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        for i, batch in enumerate(valid_loader):
            loss, acc = fast_adapt(model, batch, args.test_way, args.test_shot, args.test_query, criterion, metric = pairwise_distances_logits, device = None)
            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, n_loss/loss_ctr, n_acc/loss_ctr))
        print('###################')
        print()
        
        ## validation loss 기준으로 값 저장하기
        val_loss = n_loss/loss_ctr
        if best_loss > val_loss :
            best_loss = val_loss 
            save_file = os.path.join(f'./checkpoint/epoch{epoch}_loss{best_loss}.pth')
            torch.save(model.state_dict(), save_file)
    print('Complete Training Phase ..')
   
    # 마지막 값 저장하기
    save_file = f'./checkpoint/last_model.pth'
    torch.save(model.state_dict(), save_file)
    
    
    # 테스트 시작
    print('Start Testing Phase !! ')
    for i, batch in enumerate(test_loader, 1):
        loss, acc = fast_adapt(model, batch, args.test_way, args.test_shot, args.test_query, criterion, metric = pairwise_distances_logits, device = None)
        loss_ctr += 1
        n_acc += acc
        print('batch {}: {:.2f}({:.2f})'.format(i, n_acc/loss_ctr * 100, acc * 100))
    print('Complete Testing Phase ..')
    
    
    





if __name__ == "__main__" :
    main()