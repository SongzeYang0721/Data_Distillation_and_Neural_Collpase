# Data distillation, Neural Collapse and Interpretable AI

This research project is contributing to an NSF-CSIRO research project titled "Responsible AI for Climate Change" (GLC ID SCE200E4), and the machine learning algorithm is the foundation of this research.

The research project aims at linking the novel neural collapse with data distillation with potential application in nterpretable AI system. The research is mainly consist of 3 parts:

- Data Distillation
- Neural Collapse
- Interpretable AI

I have created this Github repository to update my research process.

## Introduction

This code aims at drawing a connenction between data distillation, neural collpase and interpretable AI. The main papers consider here are:

Data distillation
- https://github.com/SsnL/dataset-distillation
- https://github.com/google-research/google-research/tree/master/kip
- https://arxiv.org/pdf/2011.00050.pdf

Neural collpase
- https://github.com/tding1/Neural-Collapse


### Data Distillation

***The idea of distillation plays an important role in these situations by reducing the resources required for the model to be effective.***

Dataset distillation, in which a large dataset is distilled into a synthetic, smaller dataset. The original paper on data distillation: https://arxiv.org/abs/1811.10959. In this paper, we consider an alternative formulation called dataset distillation: we keep the model fixed and instead attempt to distill the knowledge from a large training dataset into a small one.

#### KIP and LS

We started by considering the latest method in data distillation:https://blog.research.google/2021/12/training-machine-learning-models-more.html.

The KIP and LS methods mentioned in the text are notable for their lack of distinction between the inner and outer loops. This means that they do not differentiate between the training process and the distillation process, resulting in a more streamlined approach to dataset distillation.

For more information on this topic, you can refer to the related papers provided:

https://openreview.net/forum?id=hXWPpJedrVP
https://openreview.net/forum?id=l-PrrQrK0QR

Survey Papers:

https://arxiv.org/abs/2301.07014
https://arxiv.org/abs/2301.04272
https://arxiv.org/abs/2301.05603

#### Connect Data Distillation with Neural collapse

Can neural collpase play a role in the data distillation? We invetigate the connection here. 

It is widely known that increase of inner loop can lead to a better performance for data distillation  trained on the neural network in the original paper. However, the latest method, namely KIP and LS, does not distinguish the inner and outer loop in the data distillation process. The real data is only used in calculating the kernel matrix. This potentially open door to methods that can train and distill the data in one go. 

We can potentially improve the data distillation by connecting it with the neural collapse. If we can fix the last layer of the neural network with the Simplex Equiangular Tight Frame (ETF) and train the neural work on the train data and distilled data (support data) and we backpropagate the gradient back to the distilled data, we should get a perfect distillation method that can reserve all the information of the training dataset.

We can calculate the simplex ETF from the number of classes of the training data and plug-in ETF for neural network training has proved to perform equally with the classical method (see NC paper).

### Neural Collapse

NeuralCollapse -- an intriguing empirical phenomenon that arises in the last-layer classifiers and features of neural networks during the terminal phase of training. As recently reported in [1], this phenomenon implies that:

(i) the class means and the last-layer classifiers all collapse to the vertices of a Simplex Equiangular Tight Frame (ETF) up to scaling, and
(ii) cross-example within-class variability of last-layer activations collapses to zero.

[1] Vardan Papyan, XY Han, and David L Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40):24652–24663, 2020.

[A Geometric Analysis of Neural Collapse with Unconstrained Features](https://arxiv.org/abs/2105.02375)

For example, our experiments demonstrate that one may set the feature dimension equal to the number of classes and fix the last-layer classifier to be a Simplex ETF for network training, which reduces memory cost by over 20% on ResNet18 without sacrificing the generalization performance.

**The universality of NC implies that the final classifier (i.e. the L-th layer) of a neural network always converges to a Simplex ETF, which is fully determined up to an arbitrary rotation and happens when K ≤ d.** Thus, based on the understandings of the last-layer features and classifiers, we show that we can substantially improve the cost efficiency on network architecture design without the sacrifice of performance, by **(i) fixing the last-layer classifier as a Simplex ETF**, and **(ii) reducing the feature dimension d = K**.

### What is ETF in the neural network?

The ETF is characterised by the columns of the matrix M (see below). It’s equal-sized angles between all classes, maximally separated up to scaling, and rotation invariance. The definition is shown below.

As the neural network collapse, the weight decay on W and H satisfy to the condition in Theorem 3.1 so that the last result corresponding to the ETF by M times M (see below).

### How do you get the ETF in the neural network/How do we plug in the ETF to the neural network?

The standard ETF can be calculated directly from matrix computation and itself is a matrix also (see above). The ETF is often empirically observed with a rotation and scaling matrix in the training of neural network with feature dimension larger than the number of classes.

## Experiment towards Neural Collapse

In particular, we'd like to see your observations (e.g. via learning curves generated by [weights and biases](https://wandb.ai/site)) of neural-collapsed CIFAR-10 ResNET across a number of settings, which can include but obviously not limited to

- different network sizes and depths, mini-batch sizes,
- other hyer-parameters, e.g. learning rates
- report things like validation accuracy, loss, and NC metrics

### Interpretable AI

A pointer:

Chenhao Tan (UChicago) gave this nice 30 min talk on explanable ML  https://www.youtube.com/watch?v=QlOuWbPECqM

One of the key papers mentioned in the talk is this one, his other papers on the topic are interesting too (e.g. CHI 2023)

Han Liu, Yizhou Tian, Chacha Chen, Shi Feng, Yuxin Chen, and Chenhao Tan. Learning Human-Compatible Representations for Case-Based Decision Support. In Proceedings of ICLR 2023.

## Environment

- CUDA 11.0
- python 3.8.3
- torch 1.6.0
- torchvision 0.7.0
- scipy 1.5.2
- numpy 1.19.1

## Experiment