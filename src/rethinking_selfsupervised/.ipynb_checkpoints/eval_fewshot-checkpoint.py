from __future__ import print_function

import os
import argparse
import socket
import time
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.mini_imagenet import ImageNet, MetaImageNet

from dataset.mnist import MNIST, MetaMNIST
from dataset.transform_cfg import transforms_options, transforms_list

from eval.meta_eval import meta_test
import torch.nn as nn
from torchvision.models import resnet50

def parse_option() :
    parser = argparse.ArgumentParser('argument for few shot evaluation')

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')

    # dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                    'CIFAR-FS', 'FC100'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', action='store_true', help='use trainval set')
    
    # models
    parser.add_argument('--transform', type=str, default='essential-BYOL/wandb/latest-run/files/essential-byol/1qtgce72/checkpoints/epoch=295-step=5326.ckpt')
    
    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # specify folder
    parser.add_argument('--model_path', type=str, default='', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='../../datasets/CIFAR-FS', help='path to data root')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=3, metavar='N',help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=5, metavar='N',help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',help='Size of test batch)')
    parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')
    opt = parser.parse_args()
    
    opt.data_aug = True
    
    return opt

    
def main() :
    
    args = parse_option()
    
    # get datasets, dataloaders
    train_trans, test_trans = transforms_options[opt.transform]

    if opt.dataset == 'cifarfs' :
        opt.data_root = '../../datasets/CIFAR-FS'
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                      train_transform=train_trans,
                                                      test_transform=test_trans),
                                         batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                         num_workers=opt.num_workers)
    elif opt.dataset == 'miniImagenet' :
        train_trans, test_trans = transforms_options['A']
        opt.data_root = '../../datasets/miniImageNet/'
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                      train_transform=train_trans,
                                                      test_transform=test_trans),
                                         batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                         num_workers=opt.num_workers)
    elif opt.dataset == 'mnist' :
        # double MNIST (for few shot learning)
        opt.transfomr = 'A'
        train_trans, test_trans = transforms_options['A']
        opt.data_root = '../../datasets/double_mnist'
        meta_testloader = DataLoader(MetaMNIST(args=opt, partition='test',
                                                      train_transform=train_trans,
                                                      test_transform=test_trans),
                                         batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                         num_workers=opt.num_workers)
    elif opt.dataset == 'sofar_test2' :
        opt.data_root = '../../datasets/sanitized_test2_v2/'
        meta_testloader = DataLoader(MetaSOFAR(args=opt, partition='test',
                                                      train_transform=None,
                                                      test_transform=None),
                                         batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                         num_workers=opt.num_workers)
    else : 
        raise NotImplementedError
    
    
    # load pretrained model
    model = resnet50()
    ckpt = torch.load(model_path)['state_dict']
    new_ckpt = dict()
    for key, value in ckpt.items() :
        if 'online_encoder.encoder' in key :
            new_ckpt[key] = value
    corrected_dict = { k.replace('online_encoder.encoder.', ''): v for k, v in new_ckpt.items() } # 제외
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.load_state_dict(corrected_dict, strict = False)

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True
        
    # get test result
    opt.test_batch_size = 32
    start = time.time()
    test_acc, test_std = meta_test(model, meta_testloader)
    test_time = time.time() - start
    print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std, test_time))
    

if __name__ == '__main__' :
    main()
        
    