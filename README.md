# Learning to Segment Using Overlap Score - the code
#### Deep Learning Project 2018-19
[![Oxford Engineering Science](https://www.eng.ox.ac.uk/images/logo.svg)](https://www.eng.ox.ac.uk/)

This repository is a submodule of my [4th year project](https://github.com/HMellor/4YP).

### Loss functions
Since the IoU surrogate I am working towards is highly non differentiable, I implemented all loss functions in hinge loss form to simplify the code and the comparisons between different loss functions.

There are three loss functions explored are as follows:
  - Micro-average
  - Macro-average
  - Overlap Score (IoU) Surrogate (Adapted from a multi-label classification algorithm written by Z. Chen)
  
![](https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_equation.png)

### Networks
I used an [fully convolutional version of AlexNet](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf), long et al., because it's small size meant I could run experiments in quick succession to validate code.

![](http://meetshah1995.github.io/images/blog/ss/fcn.png)

### Superpixels
To reduce the problem size and experiment with their effectiveness I also implemented a pipeline for performing all the experiments using superpixels. The clustering algorithm I used was [SLICO](https://ivrl.epfl.ch/research-2/research-current/research-superpixels/#SLICO), Achanta et al. 

![](https://ivrl.epfl.ch/wp-content/uploads/2018/08/156079_SLICO.jpg)

## Usage
Launch [visdom](https://github.com/facebookresearch/visdom#launch) by running (in a separate terminal window):
```
visdom -port 8098    # This is the port I used in my code, you can change it to your liking
```
Train the model like this:
```
python train.py [-h] [-lr LR] [-wd WD] [-sp SP] [-n [NAME]] [-e] [config]
```
Where each of the arguments are defined as follows:
```
positional arguments:
  config                path of configuration file to use

optional arguments:
  -h, --help                        show this help message and exit
  
  -lr LR, --learning_rate LR        learning rate to use
  
  -wd WD, --weight_decay WD         weight decay to use
  
  -sp SP, --superpixels SP          how many superpixels to use
  
  -n [NAME], --name [NAME]          name to give the experiment output directory
                                    
  -e, --evaluate                    causes prediction/image pairs to be saved for later
                                    evaluation
```
By altering the values in the lists `configs` and `sp_levels` in `big_train.py` you can automatically cross validate any combination of loss function and superpixel level. The script takes no arguments so is run as follows:
```
python big_train.py
```
_______
### The README from the original [codebase](https://github.com/meetshah1995/pytorch-semseg), for the interested:

# pytorch-semseg

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/meetshah1995/pytorch-semseg/blob/master/LICENSE)
[![pypi](https://img.shields.io/pypi/v/pytorch_semseg.svg)](https://pypi.python.org/pypi/pytorch-semseg/0.1.2)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1185075.svg)](https://doi.org/10.5281/zenodo.1185075)



## Semantic Segmentation Algorithms Implemented in PyTorch

This repository aims at mirroring popular semantic segmentation architectures in PyTorch.


<p align="center">
<a href="https://www.youtube.com/watch?v=iXh9aCK3ubs" target="_blank"><img src="https://i.imgur.com/agvJOPF.gif" width="364"/></a>
<img src="https://meetshah1995.github.io/images/blog/ss/ptsemseg.png" width="49%"/>
</p>


### Networks implemented

* [PSPNet](https://arxiv.org/abs/1612.01105) - With support for loading pretrained models w/o caffe dependency
* [ICNet](https://arxiv.org/pdf/1704.08545.pdf) - With optional batchnorm and pretrained models
* [FRRN](https://arxiv.org/abs/1611.08323) - Model A and B
* [FCN](https://arxiv.org/abs/1411.4038) - All 1 (FCN32s), 2 (FCN16s) and 3 (FCN8s) stream variants
* [U-Net](https://arxiv.org/abs/1505.04597) - With optional deconvolution and batchnorm
* [Link-Net](https://codeac29.github.io/projects/linknet/) - With multiple resnet backends
* [Segnet](https://arxiv.org/abs/1511.00561) - With Unpooling using Maxpool indices


#### Upcoming

* [E-Net](https://arxiv.org/abs/1606.02147)
* [RefineNet](https://arxiv.org/abs/1611.06612)

### DataLoaders implemented

* [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
* [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html)
* [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
* [MIT Scene Parsing Benchmark](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)
* [Cityscapes](https://www.cityscapes-dataset.com/)
* [NYUDv2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
* [Sun-RGBD](http://rgbd.cs.princeton.edu/)


### Requirements

* pytorch >=0.4.0
* torchvision ==0.2.0
* visdom >=1.0.1 (for loss and results visualization)
* scipy
* tqdm

#### One-line installation

`pip install -r requirements.txt`

### Data

* Download data for desired dataset(s) from list of URLs [here](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html#sec_datasets).
* Extract the zip / tar and modify the path appropriately in `dataset_config.json`


### Usage

Launch [visdom](https://github.com/facebookresearch/visdom#launch) by running (in a separate terminal window)

```
python -m visdom.server
```

**To train the model :**

```
python train.py [-h] [--config [CONFIG]]

--config                Configuration file to use
```

**To validate the model :**

```
usage: validate.py [-h] [--config [CONFIG]] [--model_path [MODEL_PATH]]
                       [--eval_flip] [--measure_time]

  --config              Config file to be used
  --model_path          Path to the saved model
  --eval_flip           Enable evaluation with flipped image | True by default
  --measure_time        Enable evaluation with time (fps) measurement | True
                        by default
```

**To test the model w.r.t. a dataset on custom images(s):**

```
python test.py [-h] [--model_path [MODEL_PATH]] [--dataset [DATASET]]
               [--dcrf [DCRF]] [--img_path [IMG_PATH]] [--out_path [OUT_PATH]]

  --model_path          Path to the saved model
  --dataset             Dataset to use ['pascal, camvid, ade20k etc']
  --dcrf                Enable DenseCRF based post-processing
  --img_path            Path of the input image
  --out_path            Path of the output segmap
```


**If you find this code useful in your research, please consider citing:**

```
@article{mshahsemseg,
    Author = {Meet P Shah},
    Title = {Semantic Segmentation Architectures Implemented in PyTorch.},
    Journal = {https://github.com/meetshah1995/pytorch-semseg},
    Year = {2017}
}
```
