# Pytorch U-Net <img src=images/sample_fish.png />


This repository contains a simple PyTorch implementation of an U-Net for semantic segmentation of fish images, using [this dataset](http://groups.inf.ed.ac.uk/f4k/GROUNDTRUTH/RECOG/) by B. J. Boom, P. X. Huang and J. He, R. B. Fisher [1].

Here is a sample fish image and its ground truth mask:

<img src=images/fish_000004249599_07973.png/> <img src=images/mask_000004249599_07973.png/>

The model is very simple and not super accurate, but the results are kinda cute:

<img src=images/results.png />

The code for the U-Net is partially based on this [Kaggle kernel](https://www.kaggle.com/mlagunas/naive-unet-with-pytorch-tensorboard-logging).

[[1] B. J. Boom, P. X. Huang, J. He, R. B. Fisher, "Supporting Ground-Truth annotation of image datasets using clustering", 21st Int. Conf. on Pattern Recognition (ICPR), 2012](https://ieeexplore.ieee.org/document/6460437/)
