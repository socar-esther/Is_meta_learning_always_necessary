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

# hyperparameters

## train task시에는 test에서 사용할 way, shot에 상관없이 확인
max_epoch = 200
train_shot = 50
train_way = 30
train_query = 15


## few shot task 실험시에 확인할 way, shot 확인
test_shot = 1 
test_way = 3
test_query = 15

n_gpu = 2 

# metric
def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits

# dataset
def get_datasets() :
    
    # dataset path 정의
    path_data = '../../datasets'
    train_dataset = l2l.vision.datasets.CIFARFS(root=path_data, mode='train',transform = transforms.ToTensor(), download = True)
    valid_dataset = l2l.vision.datasets.CIFARFS(root=path_data, mode='validation', transform = transforms.ToTensor(), download =True)
    test_dataset = l2l.vision.datasets.CIFARFS(root=path_data, mode='test', transform = transforms.ToTensor(), download = True)
    
    # get dataset and laoder
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
            NWays(train_dataset, train_way),
            KShots(train_dataset, train_query + train_shot), # train shot 적용
            LoadData(train_dataset),
            RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms, num_tasks = 20000)
    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)

    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
            NWays(valid_dataset, test_way),
            KShots(valid_dataset, test_query + test_shot),
            LoadData(valid_dataset),
            RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,task_transforms=valid_transforms,num_tasks=100) # 여기는 task 갯수 상관없음 (>200)
    valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)

    test_dataset = l2l.data.MetaDataset(test_dataset)
    test_transforms = [
            NWays(test_dataset, test_way),
            KShots(test_dataset, test_query + test_shot),  # test shot 적용
            LoadData(test_dataset),
            RemapLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,task_transforms=test_transforms,num_tasks=20000)
    test_loader = DataLoader(test_tasks, pin_memory=True, shuffle=True)
    
    return train_tasks, train_loader, valid_tasks, valid_loader, test_tasks, test_loader

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


if __name__ == "__main__" :
    
    # model setting (not Imagenet pretrained)
    model = resnet50(pretrained = False)
    model.cuda()
    
    # multi gpus check
    if torch.cuda.is_available():
        if n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True
    
    # dataset settings
    train_tasks, train_loader, valid_tasks, valid_loader, test_tasks, test_loader = get_datasets()
    
    # optimizer settings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    
    # 학습시작
    best_loss = 10000000.0
    print('Start Training Phase !! ')
    for epoch in range(1, max_epoch + 1):
        model.train()
        loss_ctr = 0
        n_loss = 0
        n_acc = 0

        for i in range(100):
            batch = next(iter(train_loader))
            loss, acc = fast_adapt(model, batch, train_way, train_shot, train_query, criterion, metric = pairwise_distances_logits, device = None )
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
            loss, acc = fast_adapt(model, batch, test_way, test_shot, test_query, criterion, metric = pairwise_distances_logits, device = None)
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
        loss, acc = fast_adapt(model, batch, test_way, test_shot, test_query, criterion, metric = pairwise_distances_logits, device = None)
        loss_ctr += 1
        n_acc += acc
        print('batch {}: {:.2f}({:.2f})'.format(i, n_acc/loss_ctr * 100, acc * 100))
    print('Complete Testing Phase ..')