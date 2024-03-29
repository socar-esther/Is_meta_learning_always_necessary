# Is Meta-Learning Always Necessary?: A Practical ML Framework Solving Novel Tasks at Large-scale Car Sharing Platform

Implementation of the paper submitted to WACV 2023: "OWL: Open-World Labeler for Efficient Data Acquisition in Large-scale Car-Sharing Platform"

## Paper Abstract
In real world computer vision applications, acquiring a well-labeled dataset is the first but challenging task for practitioners. The few-shot learning is studied as one presumable solution for this challenge, but we observe several hurdles to be applied in the wild. First, few-shot learning methods are not actively validated under the dataset shift, although it frequently happens in the real world. Second, while the target dataset (test set in the original few-shot learning studies) has been presumed as a closed-set setting, we claim that the real world is an open-world setting. To this end, we propose Open-World Labeler (OWL), which retrieves target samples from the unlabeled open-world dataset given a few labeled support samples. Our study illustrates our journey to apply the state-of-the-art few-shot learning method for an efficient data acquisition approach with the following contributions. First, we prove that the state-of-the-art method embraces a larger knowledge capacity than other approaches; thus, it is advantageous in understanding unseen samples. Second, we experimentally validate this approach's effectiveness under various dataset shifts with public benchmark datasets. Third, we also examine its supremacy in domain datasets acquired in the large-scale car-sharing platform. Lastly, we show the proposed OWL effectively retrieves target samples from the unlabeled open-world data pool under the live service. Based on our applicative study, we highly expect practitioners can utilize our OWL in their domains for efficient data acquisition.



## Prepare Dataset

Please visit the Github Repository (https://github.com/socar-kp/sofar_image_dataset) to check sample images utilized in this paper or acquire a full access to the dataset.

If you aim to reproduce the study, we recommend you to submit a request form to the dataset in the aforementioned Github Repository.

After you download the dataset from the given URL, please put the dataset in ./datasets directory, and named this directory 'sofar_v3'

In case of any problems or inquiries, please raise the Issue.


## How to run
### (1) Few-shot learning with representation learning
- Here is the sample commands for running the supervised learner (**ours**)  
- If you want to see the scratch notebook code, refer this : [URL](https://github.com/socar-esther/Is_meta_learning_always_necessary/blob/master/src/rethinking_supervised/supervised_script.ipynb)
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

- other few-shot option's sample commands
  <details>
  <summary>Run the self-supervised learner, BYOL with few-shot eval</summary>
  <div markdown="1">    

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

  </div>
  </details>
  <details>
  <summary>Run the metric based meta-learning algorithm, ProtoNet</summary>
  <div markdown="1">       

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

  </div>
  </details>

  <details>
  <summary>Run the optimization based meta-learning algorithm, MAML</summary>
  <div markdown="1">       

  ```shell
  $ cd src/fewshot_MAML/

  # Training phase
  $ python train_main.py --ways 3 \
                         --shots 5 \
                         --meta_lr 0.003 \
                         --fast_lr 0.5 \
                         --meta_batch_size 32 \
                         --num_iterations 50000 



  # Inference phase
  $ python eval_fewshot.py --ways 3 \
                           --shots 5 \
                           --meta_lr 0.003 \
                           --fast_lr 0.5 \
                           --meta_batch_size 32 \
                           --num_iterations 50000 
  ```

  </div>
  </details>



### (2) Zero-shot Openset Retrieval
- Sample commands for running the Zero-shot Openset Retrieval (**ours**)
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
- If you want to check the domain shift code, refer bottom of the code of [notebook](https://github.com/socar-esther/Is_meta_learning_always_necessary/blob/master/src/rethinking_supervised/supervised_script.ipynb)


## Reference
- Part of the code is from [Rethinking Few-shot Classification: A Good Embedding Is All You Need(ICLR'22)](https://arxiv.org/abs/2003.11539), the motivation of our paper
