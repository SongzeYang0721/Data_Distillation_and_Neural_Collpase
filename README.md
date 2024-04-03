# Data distillation, Neural Collapse and Interpretable AI

This research project is contributing to an NSF-CSIRO research project titled "Responsible AI for Climate Change" (GLC ID SCE200E4), and the machine learning algorithm is the foundation of this research.

The research project aims at linking the novel neural collapse with data distillation with potential application in interpretable AI system. The research is mainly consist of 3 parts:

- Data Distillation
- Neural Collapse
- Interpretable AI

I have created this Github repository to update my research process.

## Introduction

This code aims at drawing a connenction between data distillation, neural collpase and interpretable AI.

### Data Distillation

***The idea of distillation plays an important role in these situations by reducing the resources required for the model to be effective.***

Dataset distillation, in which a large dataset is distilled into a synthetic, smaller dataset. The original paper on data distillation: https://arxiv.org/abs/1811.10959. In this paper, they keep the model fixed and instead attempt to distill the knowledge from a large training dataset into a small one.

We started by considering the latest method in data distillation:https://blog.research.google/2021/12/training-machine-learning-models-more.html. The KIP and LS methods mentioned in the text are notable for their lack of distinction between the inner and outer loops. This means that they do not differentiate between the training process and the distillation process, resulting in a more streamlined approach to dataset distillation.

For more information on this topic, you can refer to the related papers provided:

- https://openreview.net/forum?id=hXWPpJedrVP
- https://openreview.net/forum?id=l-PrrQrK0QR

Survey Papers:

- https://arxiv.org/abs/2301.07014
- https://arxiv.org/abs/2301.04272
- https://arxiv.org/abs/2301.05603

### Neural Collapse

Neural collapse, emerged in the terminal phase of training(TPT), let us understand the behavior of a special class of neural network, deep classifier neural network. It has widely observed in a number of data set and model structures. Is has mainly 4 fold of meaning:

(NC1) Cross-example within-class variability of last-layer training activations collapses to zero, as the individual activations themselves collapse to their class means.

(NC2) The class means collapse to the vertices of a simplex equiangular tight frame (ETF).

(NC3) Up to rescaling, the last-layer classifiers collapse to the class means or in other words, to the simplex ETF (i.e., to a self-dual configuration).

(NC4) For a given activation, the classifier’s decision collapses to simply choosing whichever class has the closest train class mean (i.e., the nearest class center decision rule).

[A Geometric Analysis of Neural Collapse with Unconstrained Features](https://arxiv.org/abs/2105.02375)

**The universality of NC implies that the final classifier (i.e. the L-th layer) of a neural network always converges to a Simplex ETF, which is fully determined up to an arbitrary rotation and happens when K ≤ d.** Thus, based on the understandings of the last-layer features and classifiers, we show that we can substantially improve the cost efficiency on network architecture design without the sacrifice of performance, by **(i) fixing the last-layer classifier as a Simplex ETF**, and **(ii) reducing the feature dimension d = K**.

### Interpretable AI

A pointer:

Chenhao Tan (UChicago) gave this nice 30 min talk on explanable ML. One of the key papers mentioned in the talk is this one, his other papers on the topic are interesting too (e.g. CHI 2023):

- https://www.youtube.com/watch?v=QlOuWbPECqM
- https://arxiv.org/abs/2303.04809

### Connect Data Distillation with Neural collapse

Neural collapse build a new path towards the data distillation (see neural collapse). In the terminal phase of neural network training, the class means and the last-layer classifiers all collapse to the vertices of a Simplex Equiangular Tight Frame (ETF) up to scaling. In other words, the neural network has a tendency to reduce the data. This opens doors to do data distillation for us. 

### To-do list

- [ ]  Link data distillation with neural collapse:
    - [x]  Why do we plug in ETF in the neural net why does this make sense/feasible/doable?
    - [ ]  Find the best ETF on train data and the best generalized ETF on test data.
    - [ ]  Find the best reverse map that maps the data from the latent space to the data space.
- [ ]  The possible way to improve our model.

## Environment

- CUDA 12.0
- python 3.11.5
- torch 2.1.1
- torchvision 0.15.2
- scipy 1.11.1
- numpy 1.23.5