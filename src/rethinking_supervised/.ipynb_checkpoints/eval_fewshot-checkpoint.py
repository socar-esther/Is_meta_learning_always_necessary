from __future__ import print_function

import os
import argparse
import socket
import time
import sys

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import model_pool
from models.util import create_model

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.mnist import MNIST, MetaMNIST
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.sofar import SOFAR, MetaSOFAR
from dataset.transform_cfg import transforms_options, transforms_list

from util import adjust_learning_rate, accuracy, AverageMeter
from eval.meta_eval import meta_test
from eval.cls_eval import validate

from torchvision import transforms
print(torch.__version__)

def parse_option() :
    
    parser = argparse.ArgumentParser('argument for supervised training')
    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')

    # dataset
    parser.add_argument('--model', type=str, default='resnet50', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='SOFAR', choices=['miniImageNet', 'tieredImageNet',
                                                                                    'CIFAR-FS', 'FC100'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', action='store_true', help='use trainval set')
    
    # model
    parser.add_argument('--model_nm', type=str, default='multi', choices=['multi', 'imagenet'])

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # specify folder
    parser.add_argument('--model_path', type=str, default='', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='', help='path to data root')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=2, metavar='N',help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',help='Size of test batch)')
    parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')
    
    opt = parser.parse_args()
    
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'

    if opt.use_trainval:
        opt.trial = opt.trial + '_trainval'

    # set the path according to the environment
    if not opt.model_path:
        opt.model_path = './models_pretrained'
    if not opt.tb_path:
        opt.tb_path = './tensorboard'
    if not opt.data_root:
        opt.data_root = '../../datasets/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True
    
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    opt.model_name = '{}_{}_lr_{}_decay_{}_trans_{}'.format(opt.model, opt.dataset, opt.learning_rate,opt.weight_decay, opt.transform)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.adam:
        opt.model_name = '{}_useAdam'.format(opt.model_name)

    opt.model_name = '{}_trial_{}'.format(opt.model_name, opt.trial)

    opt.n_gpu = torch.cuda.device_count()
    
    return opt


def main() :
    
    opt = parse_option()
    
    # get datasets, dataloaders
    dataset_nm = 'sofar_test2' # choose : [cifarfs, miniImagenet, mnist]

    train_trans, test_trans = transforms_options[opt.transform]

    if dataset_nm == 'cifarfs' :
        opt.data_root = '../../datasets/CIFAR-FS'
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                      train_transform=train_trans,
                                                      test_transform=test_trans),
                                         batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                         num_workers=opt.num_workers)
    elif dataset_nm == 'miniImagenet' :
        train_trans, test_trans = transforms_options['A']
        opt.data_root = '../../datasets/miniImageNet/'
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                      train_transform=train_trans,
                                                      test_transform=test_trans),
                                         batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                         num_workers=opt.num_workers)
    elif dataset_nm == 'mnist' :
        # double MNIST (for few shot learning)
        opt.transfomr = 'A'
        train_trans, test_trans = transforms_options['A']
        opt.data_root = '../../datasets/double_mnist'
        meta_testloader = DataLoader(MetaMNIST(args=opt, partition='test',
                                                      train_transform=train_trans,
                                                      test_transform=test_trans),
                                         batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                         num_workers=opt.num_workers)
    elif dataset_nm == 'sofar_test2' :
        opt.data_root = '../../datasets/sanitized_test2_v2/'
        meta_testloader = DataLoader(MetaSOFAR(args=opt, partition='test',
                                                      train_transform=None,
                                                      test_transform=None),
                                         batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                         num_workers=opt.num_workers)
    else : 
        raise NotImplementedError
        
    # get pretrained model
    if opt.model_nm == 'imagenet' : 
        n_cls = 11
        model = create_model(opt.model, n_cls, opt.dataset)
        model = model.cuda()
        print('Experiment with Imagenet weight frozen')
    elif opt.model_nm == 'multi' :
        n_cls = 11
        #model_path = './checkpoint/ckpt_epoch_198.pth' # multi class classification
        model_path = './checkpoint/best_model_sofar.pth' # Distill multi class classification
        model = create_model(opt.model, n_cls, opt.dataset)
        ckpt = torch.load(model_path)
        corrected_dict = { k.replace('features.', ''): v for k, v in ckpt.items() } 
        model.load_state_dict(corrected_dict, strict = False)
        model = model.cuda()
        print(f'>> model name : {opt.model}, model_path : {model_path}')
        
    
    # start eval
    test_acc, test_std = meta_test(model, meta_testloader)
    print(f'Test Accuracy : {test_acc}, Test Std : {test_std}')
    

if __name__ == '__main__' : 
    main()