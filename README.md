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


## For lifelong classification

1. run `classConditionGANMemory.py` for each task until the whole sequeence of tasks are remembered and save the generators;

2. run `Lifelong_classification.py` to get the classification results.

3. run `Compression_low_rank_six_butterfly.py` to get the compression results.


Note, for the sake of simplicity, we devide the pseudo rehearsal based lifelong classification processes into above two stages, one can of course find a way to merge these two stages to form a learning process along task sequence.


## Acknowledgement
Our code is based on GAN_stability: https://github.com/LMescheder/GAN_stability from the paper [Which Training Methods for GANs do actually Converge?](https://avg.is.tuebingen.mpg.de/publications/meschedericml2018).
