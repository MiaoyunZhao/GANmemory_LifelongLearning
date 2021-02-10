# GAN Memory for Lifelong learning
This is a pytorch implementation of the NeurIPS paper [GAN Memory with No Forgetting](https://papers.nips.cc/paper/2020/file/bf201d5407a6509fa536afc4b380577e-Paper.pdf).

Please consider citing our paper if you refer to this code in your research.
```
@article{cong2020gan,
  title={GAN Memory with No Forgetting},
  author={Cong, Yulai and Zhao, Miaoyun and Li, Jianqiao and Wang, Sijia and Carin, Lawrence},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

# Requirement
```
python=3.7.3
pytorch=1.2.0
```

# Notes
The source model is based on the GP-GAN.

`GANMemory_Flowers.py` is the implementation of the model in Figure1(a).

`classConditionGANMemory.py` is the class-conditional generalization of GAN memory, which is used as pseudo rehearsal for a lifelong classification as shown in Section 5.2.

`Lifelong_classification.py` is the code for the lifelong classification part as shown in Section 5.2.

# Usage

First, download the pretrained GP-GAN model by running `download_pretrainedGAN.py`. Note please change the path therein.

Second, download the training data to the folder `./data/`. For example, download the Flowers dataset from: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ to the folder `./data/102flowers/`.


## Dataset preparation
```angular2
data
├──102flowers
           ├──all8189images
├── CelebA
...
```

Finally, run `GANMemory_Flowers.py`.

The FID scores of our method shown in Figure 1(b) are summerized in the following table.

| Dataset      |   5K  |   10K |   15K |  20K  |  25K  |  30K  |  35K  |  40K  |  45K  |  50K  |  55K  |  60K  |
| :---         |  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |
| Flowers      | 29.26 | 23.25 | 19.73 | 17.98 | 17.04 | 16.10 | 15.93 | 15.38 | 15.33 | 14.96 | 15.19 | 14.75 |
| Cathedrals   | 19.78 | 18.32 | 17.10 | 16.47 | 16.15 | 16.33 | 16.08 | 15.94 | 15.78 | 15.60 | 15.64 | 15.67 |
| Cats         | 38.56 | 25.74 | 23.14 | 21.15 | 20.80 | 20.89 | 19.73 | 19.88 | 18.69 | 18.57 | 17.57 | 18.18 |


## For lifelong classification

1. run `classConditionGANMemory.py` for each task until the whole sequeence of tasks are remembered and save the generators;

2. run `Lifelong_classification.py` to get the classification results.

3. run `Compression_low_rank_six_butterfly.py` to get the compression results.


Note, for the sake of simplicity, we devide the pseudo rehearsal based lifelong classification processes into above two stages, one can of course find a way to merge these two stages to form a learning process along task sequence.


## Acknowledgement
Our code is based on GAN_stability: https://github.com/LMescheder/GAN_stability from the paper [Which Training Methods for GANs do actually Converge?](https://avg.is.tuebingen.mpg.de/publications/meschedericml2018).
