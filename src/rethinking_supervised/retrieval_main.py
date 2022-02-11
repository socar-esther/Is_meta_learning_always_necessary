from __future__ import print_function

import os
import argparse
import shutil

import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.models import resnet50

from models import model_pool
from models.util import create_model
from scipy.spatial import distance


def parse_option() :
    parser = argparse.ArgumentParser('argument for zero-shot openset retrieval')
    
    # dataset
    parser.add_argument('--support_set_dir', type=str, default='../../datasets/open_set_few_shot_retrieval_set/support_document')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query_set_dir', type=str, default='../../datasets/open_set_few_shot_retrieval_set/query_set')
    parser.add_argument('--query_set_batch', type=int, default=1)
    parser.add_argument('--task_nm', type=str, default='document', choices=['document', 'snow'], help='The task name of the retrieval, especially the name of the support set class')
    
    
    # model
    parser.add_argument('--model_path', type=str, default='./checkpoint/best_model_sofar_multi.pth')
    parser.add_argument('--model_nm', type=str, default='resnet50')
    parser.add_argument('--n_cls', type=int, default=11, help='The number of classes in the training set')
    parser.add_argument('--data_nm', type=str, default='SOFAR', help='The name of the training dataset to create the model')
    
    # distance
    parser.add_argument('--distance_opt', type=str, default='euclidean', choices=['euclidean', 'cosine'])
    
    
    opt = parser.parse_args()
    
    return opt
    
    
    
    
def get_support_set_loader(opt) :
    normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
    transform_train = transforms.Compose([
                        transforms.Resize((550, 550)),
                        transforms.RandomCrop(448, padding=8),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize
                    ])
    support_dataset = ImageFolder(root=opt.support_set_dir, transform=transform_train)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=opt.shot, shuffle=True, num_workers=opt.num_workers)
    
    return support_dataset, support_loader

def get_query_set_loader(opt) :
    normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
    transform_train = transforms.Compose([
                        transforms.Resize((550, 550)),
                        transforms.RandomCrop(448, padding=8),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize
                    ])
    query_dataset = ImageFolder(root=opt.query_set_dir, transform=transform_train)
    query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=opt.query_set_batch, shuffle=True, num_workers=opt.num_workers)
    
    return query_dataset, query_loader


def get_pretrinaed_model(opt) :
    model = create_model(opt.model_nm, opt.n_cls, opt.data_nm)
    ckpt = torch.load(opt.model_path)
    corrected_dict = { k.replace('features.', ''): v for k, v in ckpt.items() } 
    model.load_state_dict(corrected_dict, strict = False)
    model = model.cuda()
    model.classifier = nn.Identity()
    
    return model

def main() :
    
    opt = parse_option()
    
    # get the dataloaders and pretrained model
    support_dataset, support_loader = get_support_set_loader(opt)
    query_dataset, query_loader = get_query_set_loader(opt)
    model = get_pretrinaed_model(opt)
    
    # get support set vector
    for idx, (img, target) in enumerate(support_loader) : 
        img = img.cuda()
        target = target.cuda()
        support_output = model(img)
        break
        
    # start retrieval
    result_dict = dict()

    for idx, (img, target) in enumerate(query_loader) : 
        img = img.cuda()
        target = target.cuda()
        img_path = query_dataset.imgs[idx][0]

        query_vec = model(img)
        query_vec = query_vec.cpu().detach().numpy()

        distance_list = list()
        for i in range(len(support_output)) :
            support_vec = support_output[i].reshape(1, -1)
            support_vec = support_vec.cpu().detach().numpy()
            if opt.distance_opt == 'euclidean' :
                dist = np.linalg.norm(support_vec - query_vec) # euclidean distacne
            elif opt.distance_opt == 'cosine' :
                dist = distance.cosine(support_vec, query_vec) # cosine distance
            else :
                raise NotImplementedError()
            
            distance_list.append(dist)
        avg_dist = np.mean(distance_list)
        result_dict[img_path] = avg_dist
    
    sorted_dict = sorted(result_dict.items(), key = lambda item: item[1])
    sorted_dict_50 = sorted_dict[:50]  # top 50
    
    top_50_list = list()

    for key, item in sorted_dict_50 : 
        top_50_list.append(key)
        
    count = 0

    for i in range(len(top_50_list)) :
        if opt.task_nm in top_50_list[i] :
            count += 1
    
    print('@50 Accuracy=', count / 50)
    

if __name__ == '__main__' :
    main()