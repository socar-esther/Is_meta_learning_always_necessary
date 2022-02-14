# Is meta-learning always necessary
Implementation of KDD 2022 "Is Meta-Learning Always Necessary?: A Practical ML Framework Solving Novel Tasks at Large-scale Car Sharing Platform"

## Datasets
- We open our domain benchmark set in https://socar-kp.github.io/sofar_image_dataset/
- Download this dataset in ./datasets directory, and named this directory 'sofar_v3'

## How to run
### Few-shot learning
- Run the supervised learner (**ours**)
```shell
$ cd src/rethinking_supervised/

# Training phase
$ python train_main.py --batch_size 4 \
                       --learning_rate 1e-5 \
                       --lr_decay_epochs 60,80 \
                       --model resnet50 \
                       --dataset SOFAR 

# Inference phase          
$ python eval_fewshot.py --batch_size 4 \
                         --epochs 200 \
                         --learning_rate 1e-3 \
                         --model resnet50 \
                         --dataset SOFAR \
                         --model_nm multi \
                         --n_ways 3 \
                         --n_shots 5 \
                         --data_root ../../datasets/sanitized_test2_v2/  
```
- Run the self-supervised learner
```shell
$ cd src/rethinking_selfsupervised/

# Training phase
$ python train_main.py \
    --gpus 4 \
    --distributed_backend ddp \
    --sync_batchnorm \
    --dataset SOFAR \
    --batch_size 128 \
    --max_epochs 1000 \
    --arch resnet50 \
    --precision 16 \
    --comment wandb-comment

# Inference phase
$ python eval_fewshot.py --batch_size 4 \
                         --epochs 200 \
                         --learning_rate 1e-3 \
                         --model resnet50 \
                         --dataset SOFAR \
                         --model_nm multi \
                         --n_ways 3 \
                         --n_shots 5 \
                         --data_root ../../datasets/sanitized_test2_v2/  
```
- Run the metric based meta-learning algorithm, ProtoNet
```shell
$ cd src/fewshot_Protonet/

# Training phase
$ python train_main.py --max_epoch 200 \
                       --train_shot 10 \
                       --train_way 3 \
                       --train_query 15 \
                       --test_shot 5 \
                       --test_way 3 \
                       --test_query 15 \
                       --n_gpu 4 
                       

# Inference phase
$ python eval_fewshot.py --test_shot 5 \
                         --test_way 3 \
                         --test_query 15 \
                         --dataset_nm cifarfs \
                         --model_path ./checkpoint/epoch50_loss1.414059302210808.pth \
                         --n_gpu 2 

```
- Run the optimization based meta-learning algorithm, MAML
```shell
$ cd src/fewshot_MAML/

# Training phase
$


# Inference phase

$
```

### Zero-shot Openset Retrieval
- Run the Zero-shot Openset Retrieval (**ours**)
```shell
$ cd src/rethinking_supervised/
$ python retrieval_main.py  --support_set_dir ../../datasets/open_set_few_shot_retrieval_set/support_document \
                            --query_set_dir ../../datasets/open_set_few_shot_retrieval_set/query_set \
                            --shot 5 \
                            --task_nm document \
                            --model_path ./checkpoint/best_model_sofar_multi.pth \
                            --model_nm resnet \
                            --n_cls 11 \
                            --data_nm SOFAR \
                            --distance_opt euclidean
```
- If you want to check the domain shift code, refer bottom of the code of [notebook](https://github.com/socar-esther/Is_meta_learning_always_necessary/blob/master/src/rethinking_supervised/scratch_notebook/supervised_script.ipynb)


## Acknowlegements
- Part of the code is from [github](https://github.com/WangYueFt/rfs), the motivation of our paper
