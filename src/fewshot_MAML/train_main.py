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

    # Create Tasksets using the benchmark interface
    tasksets = l2l.vision.benchmarks.get_tasksets('cifarfs',
                                                  train_samples=2*args.shots,
                                                  train_ways=args.ways,
                                                  test_samples=2*args.shots,
                                                  test_ways=args.ways,
                                                  num_tasks=100000,
                                                  root='../../datasets',
    )

    # Create model
    model = resnet50(pretrained=False) 
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=True)
    opt = optim.Adam(maml.parameters(), args.meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')
    best_acc = 0.0
    
    for iteration in range(args.num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(args.meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               args.adaptation_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # # Compute meta-validation loss
            # learner = maml.clone()
            # batch = tasksets.validation.sample()
            # evaluation_error, evaluation_accuracy = fast_adapt_test(batch,
            #                                                    learner,
            #                                                    loss,
            #                                                    adaptation_steps,
            #                                                    shots,
            #                                                    ways,
            #                                                    device)
            # meta_valid_error += evaluation_error.item()
            # meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / args.meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / args.meta_batch_size)
        # print('Meta Valid Error', meta_valid_error / meta_batch_size)
        # print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / args.meta_batch_size)
        opt.step()
        
        # save the weight
        train_acc = meta_train_accuracy / args.meta_batch_size
        if train_acc > best_acc : 
            best_acc = train_acc
            save_file = f'./checkpoint/iter{iteration}_acc{train_acc}.pth'
            torch.save(model.state_dict(), save_file)


if __name__ == '__main__':
    main()
