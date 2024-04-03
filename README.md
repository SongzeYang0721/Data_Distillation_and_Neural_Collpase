# Data distillation, Neural Collapse and Interpretable AI

This research project is contributing to an NSF-CSIRO research project titled "Responsible AI for Climate Change" (GLC ID SCE200E4), and the machine learning algorithm is the foundation of this research.

The research project aims at linking the novel neural collapse with data distillation with potential application in nterpretable AI system. The research is mainly consist of 3 parts:

- Data Distillation
- Neural Collapse
- Interpretable AI

I have created this Github repository to update my research process.

## Environment

- CUDA 12.0
- python 3.11.5
- torch 2.1.1
- torchvision 0.15.2
- scipy 1.11.1
- numpy 1.23.5

## Introduction

This code aims at drawing a connenction between data distillation, neural collpase and interpretable AI.

### Data Distillation

***The idea of distillation plays an important role in these situations by reducing the resources required for the model to be effective.***

Dataset distillation, in which a large dataset is distilled into a synthetic, smaller dataset. The original paper on data distillation: https://arxiv.org/abs/1811.10959. In this paper, we consider an alternative formulation called dataset distillation: we keep the model fixed and instead attempt to distill the knowledge from a large training dataset into a small one.

We started by considering the latest method in data distillation:https://blog.research.google/2021/12/training-machine-learning-models-more.html.

The KIP and LS methods mentioned in the text are notable for their lack of distinction between the inner and outer loops. This means that they do not differentiate between the training process and the distillation process, resulting in a more streamlined approach to dataset distillation.

For more information on this topic, you can refer to the related papers provided:

- https://openreview.net/forum?id=hXWPpJedrVP
- https://openreview.net/forum?id=l-PrrQrK0QR

Survey Papers:

- https://arxiv.org/abs/2301.07014
- https://arxiv.org/abs/2301.04272
- https://arxiv.org/abs/2301.05603

### Neural Collapse

NeuralCollapse -- an intriguing empirical phenomenon that arises in the last-layer classifiers and features of neural networks during the terminal phase of training. As recently reported in [1], this phenomenon implies that:

(i) the class means and the last-layer classifiers all collapse to the vertices of a Simplex Equiangular Tight Frame (ETF) up to scaling, and
(ii) cross-example within-class variability of last-layer activations collapses to zero.

[1] Vardan Papyan, XY Han, and David L Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40):24652–24663, 2020.

[A Geometric Analysis of Neural Collapse with Unconstrained Features](https://arxiv.org/abs/2105.02375)

**The universality of NC implies that the final classifier (i.e. the L-th layer) of a neural network always converges to a Simplex ETF, which is fully determined up to an arbitrary rotation and happens when K ≤ d.** Thus, based on the understandings of the last-layer features and classifiers, we show that we can substantially improve the cost efficiency on network architecture design without the sacrifice of performance, by **(i) fixing the last-layer classifier as a Simplex ETF**, and **(ii) reducing the feature dimension d = K**.

### Interpretable AI

A pointer:

Chenhao Tan (UChicago) gave this nice 30 min talk on explanable ML  https://www.youtube.com/watch?v=QlOuWbPECqM

One of the key papers mentioned in the talk is this one, his other papers on the topic are interesting too (e.g. CHI 2023)

Han Liu, Yizhou Tian, Chacha Chen, Shi Feng, Yuxin Chen, and Chenhao Tan. Learning Human-Compatible Representations for Case-Based Decision Support. In Proceedings of ICLR 2023.

#### Connect Data Distillation with Neural collapse

Can neural collpase play a role in the data distillation? We invetigate the connection here. 

It is widely known that increase of inner loop can lead to a better performance for data distillation  trained on the neural network in the original paper. However, the latest method, namely KIP and LS, does not distinguish the inner and outer loop in the data distillation process. The real data is only used in calculating the kernel matrix. This potentially open door to methods that can train and distill the data in one go. 

We can potentially improve the data distillation by connecting it with the neural collapse. If we can fix the last layer of the neural network with the Simplex Equiangular Tight Frame (ETF) and train the neural work on the train data and distilled data (support data) and we backpropagate the gradient back to the distilled data, we should get a perfect distillation method that can reserve all the information of the training dataset.

We can calculate the simplex ETF from the number of classes of the training data and plug-in ETF for neural network training has proved to perform equally with the classical method (see NC paper).

## Experiment towards Neural Collapse