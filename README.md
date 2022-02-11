# Is_meta_learning_always_necessary
Implementation of KDD 2022 "Is Meta-Learning Always Necessary?: A Practical ML Framework Solving Novel Tasks at Large-scale Car Sharing Platform"

## Datasets
- We open our domain benchmark set in https://socar-kp.github.io/sofar_image_dataset/

## How to run
- Run the supervised learner (ours)
```shell
$ cd src/rethinking_supervised/

# Training phase
$ python train_main.py --batch_size 4 \
                       --epochs 500 \
                       --learning_rate 1e-3 \
                       --model resnet50 \
                       --dataset SOFAR \
                       --n_test_runs 600 \
                       --n_ways 3 \ 
                       --n_shots 5 \
                       --data_root ../../datasets/sofar_v3/ 
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
- Run the Zero-shot Openset Retrieval (ours)
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
