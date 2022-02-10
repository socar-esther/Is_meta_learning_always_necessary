import random
import numpy as np
from argparse import ArgumentParser

import torch
from torch import nn, optim
from torchvision.models import resnet50

from utils import *
import learn2learn as l2l
from learn2learn.data.transforms import (NWays,
                                         KShots,
                                         LoadData,
                                         RemapLabels,
                                         ConsecutiveLabels)



def parse_option() :
    parser = ArgumentParser()
    parser.add_argument('--ways', default=3, type=int)
    parser.add_argument('--shots', default=5, type=int)
    parser.add_argument('--meta_lr', default=0.003, type=float)
    parser.add_argument('--fast_lr', default=0.5, type=float)
    parser.add_argument('--meta_batch_size', default=32, type=int)
    parser.add_argument('--adaptation_steps', default=1, type=int)
    parser.add_argument('--num_iterations', default=60000, type=int)
    opt = parser.parse_args()
    return opt
    
    
    

def main():
    cuda=True,
    seed=42
    
    args = parse_option()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
        
    # get test tasksets
    tasksets = l2l.vision.benchmarks.get_tasksets('mini-imagenet',
                                                  train_samples=2*args.shots,
                                                  train_ways=args.ways,
                                                  test_samples=2*args.shots,
                                                  test_ways=args.ways,
                                                  root='../../datasets')
    
    meta_test_error = 0.0
    meta_test_accuracy = 0.0

    for task in range(args.meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch, learner, loss, adaptation_steps, args.shots, args.ways, device)

        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()

    print('Meta Test Error', meta_test_error / args.meta_batch_size)
    print('Meta Test Accuracy', meta_test_accuracy / args.meta_batch_size)