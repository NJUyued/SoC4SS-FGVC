# SoC4SS-FGVC

Code for CVPR 2023 Submission (ID: 8985) "*Roll With the Punches: Expansion and Shrinkage of Soft Label Selection for Semi-Supervised Fine-Grained Learning*".

## Requirements

- numpy==1.20.3
- Pillow==9.3.0
- scikit_learn==1.1.3
- torch==1.8.0
- torchvision==0.9.0


## How to Train
### Important Args
- `--net [resnet18/resnet50/resnet101/wrn/wrnvar/preresnet/cnn13]`: By default, ResNet-50 is used for experiments.  We provide alternatives as follows: ResNet-18/101, Wide ResNet, Wide ResNet-Var (WRN-37-2), PreAct ResNet and CNN-13.
- `--dataset [semi-fungi/semi-aves]` and `--data_dir`: Your dataset name and path. We support two datasets: Semi-Fungi and Semi-Aves (See "*A Realistic Evaluation of Semi-Supervised Learning for Fine-Grained Classification*" for details).
- `--unlabel [in/inout]`: The unlabeled data used for training. The unlabeled data of Semi-Fungi and Semi-Aves contains two subsets. One has in-distribution data only, the other one has both in-distribution and out-of-distribution data.
- `--alpha`: $\alpha$ for confidence-aware k selection.
- `--num_tracked_batch`: $N_{B}$ for class transition tracking (CTT).
- `--resume`, `--load_path`, `--load_path_soc` and `--pretrained`: 
    - If you want to resume training without `soc.pkl`, i.e., saved data of `centroids` (for CTT-based k-means clustering) and `label_matrix` (for CTT), please set `--resume --load_path@path to your checkpoint`. 
    - If you want to resume training with `soc.pkl`, please set `--resume --load_path@path to your checkpoint@ --load_path_soc @path to your soc.pkl@`. 
    - If you want to use the expert model or MoCo model, please set `--resume --load_path @path to expert model/MoCo model@ --pretrained`.

### Training with Single GPU

```
python train_soc.py --rank 0 --gpu [0/1/...] @@@other args@@@
```
### Training with Multi-GPUs

- Using DataParallel

```
python train_soc.py --world-size 1 --rank 0 @@@other args@@@
```

- Using DataParallel DistributedDataParallel with single node

```
python train_soc.py --world-size 1 --rank 0 --multiprocessing-distributed @@@other args@@@
```

## Examples of Running

To better reproduce our experimental results, it is recommended to use multi-GPUs with DataParallel for training.

### Using In-distribuion Unlabeleda Data 
#### Training from scratch

```
python train_soc.py --world-size 1 --rank 0 --seed 1 --num_eval_iter 2000 --overwrite --save_name aves_in_sc --dataset semi_aves --num_classes 200 --unlabel in 
```

#### Training from scratch with MoCo

```
python train_soc.py --world-size 1 --rank 0 --seed 1 --num_eval_iter 1000 --overwrite --save_name aves_in_sc_moco --dataset semi_aves --num_classes 200 --unlabel in --resume --load_path @path to MoCo pre-trained model@ --pretrained --num_train_iter 200000
```

#### Training from expert or expert with MoCo

```
python train_soc.py --world-size 1 --rank 0 --seed 1 --num_eval_iter 500 --overwrite --save_name aves_in_pre --dataset semi_aves --num_classes 200 --unlabel in --resume --load_path @path to pre-trained model@ --pretrained --lr 0.001 --num_train_iter 50000
```


The expert models and MoCo models can be obtained [here][ck] (provided by https://github.com/cvl-umass/ssl-evaluation).


### Using Out-of-Distribuion Unlabeleda Data 
#### Training from scratch

```
python train_soc.py --world-size 1 --rank 0 --seed 1 --num_eval_iter 2000 --overwrite --save_name aves_inout_sc --dataset semi_aves --num_classes 200 --unlabel inout 
```

***
## Evaluation
Each time you start training, the evaluation results of the current model will be displayed. If you want to evaluate a model, use its checkpoints to resume training, i.e., use `--resume --load_path load_path@path to your checkpoint@`.

## Results (e.g. seed=1)

| Dateset | Unlabeled Data | Pre-training | Top1-Acc (%)| Top5-Acc (%)| Checkpoint |
| :-----:| :----: | :----: |:----: |:----: |:----: |
|Semi-Aves | in-distribution | - | 32.3 | 55.5 | [here][av-in-sc] |
| | | MoCo | 39.5 | 62.5 | [here][av-in-sc-mc] |
| | | ImageNet | 56.8 | 79.1 | [here][av-in-im] |
| | | ImageNet  + MoCo | 57.1 | 79.1 | [here][av-in-im-mc] |
| | | iNat | 71.0 | 88.4 | [here][av-in-in] |
| | | iNat + MoCo | 70.2 | 88.3 | [here][av-in-in-mc] |
| | out-of-distribution | - | 27.5 | 50.7 | [here][av-inout-sc] |
| |  | MoCo | 40.4 | 65.9 | [here][av-inout-sc-mc] |
|Semi-Fungi | in-distribution | - | 38.50 | 61.35 | [here][fg-in-sc] |
| | | MoCo | 46.9 | 71.4 | [here][fg-in-sc-mc] |
| | | ImageNet | 61.1 | 83.2 | [here][fg-in-im] |
| | | ImageNet  + MoCo | 61.8 | 85.9 | [here][fg-in-im-mc] |
| | | iNat | 62.3 | 85.0 | [here][fg-in-in] |
| | | iNat + MoCo | 62.2 | 84.4 | [here][fg-in-in-mc] |
| | out-of-distribution | - | 35.6 | 60.6 | [here][fg-inout-sc] | 
| |  | MoCo | 50.0 | 74.8 | [here][fg-inout-sc-mc] |

[av-in-im]: https://drive.google.com/drive/folders/1apctbIN_O9EuD8ZXrr7Diwq-1z0qADBu?usp=share_link
[av-in-im-mc]: https://drive.google.com/drive/folders/1lx-DYwCF1bDdGoy5nUQ_0Kdp6jhwQ_qi?usp=share_link
[av-in-in]: https://drive.google.com/drive/folders/1C4RcpnSmWcwpbSpkAicbcdjx-7Su80Rb?usp=share_link
[av-in-in-mc]: https://drive.google.com/drive/folders/1NC9HCB1sdbPhEd3SeStfrVQEBLTPKc_A?usp=share_link
[av-in-sc]: https://drive.google.com/drive/folders/1ML3WJeH20achx5KxZQYGCVrRNCAfJl0Y?usp=share_link
[av-in-sc-mc]: https://drive.google.com/drive/folders/1dyY-ylLI0op0-MiKFpYhfctTFp5iPUpJ?usp=share_link
[av-inout-sc]: https://drive.google.com/drive/folders/105EDpTelNa7oURIV0TGpHKf2W80M32yN?usp=share_link
[av-inout-sc-mc]: https://drive.google.com/drive/folders/1gy9KSXJ4OX3SKJtf8N9aritxP12atVkJ?usp=share_link
[fg-in-im]: https://drive.google.com/drive/folders/1cn8QTJFJnDhlgR-vDHaJNM5_NaBQsbAB?usp=share_link
[fg-in-im-mc]: https://drive.google.com/drive/folders/1Ug4C9qpTmL_H3760gTt0FfSpGPj6tpuS?usp=share_link
[fg-in-in]: https://drive.google.com/drive/folders/1DygNzUCNc9BFhhK2rmy7JkRoOb-x2SNP?usp=share_link
[fg-in-in-mc]: https://drive.google.com/drive/folders/13kvItyHZqZiL8AAViuU5Y46CpXMfPMSw?usp=share_link
[fg-in-sc]: https://drive.google.com/drive/folders/15s-upb33Uo1_dpF9xLQSkvgu4MAsCBQA?usp=share_link
[fg-in-sc-mc]: https://drive.google.com/drive/folders/1Sk5E9H5J8QaslyIxH2HilUWK8NEmiG-b?usp=share_link
[fg-inout-sc]: https://drive.google.com/drive/folders/1EO8IHoO8TW9YhWKO-3bv9iCj2Mya0hXy?usp=share_link
[fg-inout-sc-mc]: https://drive.google.com/drive/folders/1CW8QwusyAlF2kL6zT94lUgx5AIxgMSQg?usp=share_link
[ck]: http://vis-www.cs.umass.edu/semi-inat-2021/ssl_evaluation/models/