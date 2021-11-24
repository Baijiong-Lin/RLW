# RLW
This repository contains the source code of Random Loss Weighting (RLW) from our paper, A Closer Look at Loss Weighting in Multi-Task Learning.

## Note
The implementation of baseline methods (state-of-the-art weighting strategies in multi-task learning) in our paper will be released soon.

## Environment

- Python 3.7.10
- torch 1.8.0+cu111
- torchvision 0.9.0+cu111
- scipy 1.2.1
- numpy 1.20.2
- transformers 4.6.1



## Preparing Data

| Datasets                                         | How to get it?                                               | Comments                                                     |
| ------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| NYUv2                                            | Download from [here](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0) (288x384, 8.4G) | Pre-processed by [mtan](https://github.com/lorenmt/mtan)     |
| CityScapes                                       | Download from [here](https://www.dropbox.com/sh/gaw6vh6qusoyms6/AADwWi0Tp3E3M4B2xzeGlsEna?dl=0) (128x256, 6.1G) | Pre-processed by [mtan](https://github.com/lorenmt/mtan)     |
| Office-31                                        | Download from [here](https://www.cc.gatech.edu/~judy/domainadapt/#datasets_code) (88M) |                                                              |
| Office-Home                                      | Download from [here](https://www.hemanthdv.org/officeHomeDataset.html) (1.2G) |                                                              |
| CelebA                                           | Download from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (23G) |                                                              |
| PASCAL-Context                                   | Download from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (3.6G) | Referenced from [astmt](https://github.com/facebookresearch/astmt) |
| Four multilingual problems from XTREME benchmark | Download Wikiann dataset from [here](https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN) for NER,  run `sh ./XTREME/propocess_data/download_data.sh` to automatically download the other datasets and pre-pocessing | Referenced from [xtreme](https://github.com/google-research/xtreme) |



## Experiments

The implementation details of all datasets are described as follow,

| Dataset                                          | Backbone                                                     | Main File Name                         | Flags                                                        | Comments                                                     |
| ------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| NYUv2                                            | DeepLabV3+ with ResNet-50 (Referenced from [mtan](https://github.com/lorenmt/mtan)) | `./nyu_cityscapes/train_nyu.py`        | data_root, gpu_id, weighting, random_distribution, model, aug | model: DMTL, MTAN (Official implementation in [mtan](https://github.com/lorenmt/mtan), Cross_Stitch (Unofficial implementation by us), NDDRCNN (Official implementation in [Multi-Task-Learning-PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch)) |
| CityScapes                                       | DeepLabV3+ with ResNet-50 (Referenced from [mtan](https://github.com/lorenmt/mtan)) | `./nyu_cityscapes/train_cityscapes.py` | data_root, gpu_id, weighting, random_distribution, aug       |                                                              |
| Office-31 and Office-Home                        | ResNet-18                                                    | `./office/train_office.py`             | data_root, gpu_id, weighting, random_distribution, dataset   | dataset: office-31, office_home                              |
| CelebA                                           | ResNet-18 without pretrained (Referenced from [MultiObjectiveOptimization](https://github.com/isl-org/MultiObjectiveOptimization)) | `./celeba/train_celeba.py`             | data_root, gpu_id, weighting, random_distribution            |                                                              |
| PASCAL-Context                                   | DeepLabV3+ with ResNet-18 (Referenced from [mtan](https://github.com/lorenmt/mtan) and [Multi-Task-Learning-PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch)) | `./PASCAL/train_pascal.py`             | data_root, gpu_id, weighting, random_distribution            |                                                              |
| Four multilingual problems from XTREME benchmark | mBert (Referenced from [xtreme](https://github.com/google-research/xtreme)) | `./XTREME/train_multilingual.py`       | gpu_id, weighting, random_distribution, dataset              | dataset: udpos, panx, xnli, pawsx                            |

For the flags of `weighting` and `random_distribution`, they represent

| Flag Name             | Option                                                       | Comments     |
| --------------------- | ------------------------------------------------------------ | ------------ |
| `weighting`           | EW, RLW                                                      |              |
| `random_distribution` | normal, uniform, random_normal, dirichlet, Bernoulli, Bernoulli_1 | For RLW only |

To train on any datasets, `cd DATASETNAME` first and then simple run `python -u train_DATASETNAME.py --FLAG_NAME 'FLAG_OPTION'`.


## Citation

If you found this code/work to be useful in your own research, please considering citing the following:

```latex
@article{lin2021rlw,
  title={A Closer Look at Loss Weighting in Multi-Task Learning},
  author={Lin, Baijiong and Ye, Feiyang and Zhang, Yu},
  journal={arXiv preprint arXiv:2111.10603},
  year={2021}
}
```


## Acknowledgement

We would like to thank the authors that release the public repositories as follow (listed in no particular order):  [mtan](https://github.com/lorenmt/mtan), [astmt](https://github.com/facebookresearch/astmt), [MultiObjectiveOptimization](https://github.com/isl-org/MultiObjectiveOptimization), [Pytorch-PCGrad](https://github.com/WeiChengTseng/Pytorch-PCGrad), [Multi-Task-Learning-PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch), and [xtreme](https://github.com/google-research/xtreme).
