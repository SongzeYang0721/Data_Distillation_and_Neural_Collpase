import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from data.FilteredDataset import FilteredDataset, SubsetDataset

class CIFAR10RandomLabels(CIFAR10):
    # Part from https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py
    """CIFAR10 dataset, with support for randomly corrupt labels.
    ######## Need to generate a set of all randomed label first #########
    ### Check for generate_random_label.py for an example ###
    """
    def __init__(self, **kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        if self.train:
            with open(self.root+'/cifar10_random_label/train_label.pkl', 'rb') as f:
                train_all = pickle.load(f)
                self.targets = train_all["label"]
        else:
            with open(self.root+'/cifar10_random_label/test_label.pkl', 'rb') as f:
                test_all = pickle.load(f)
                self.targets = test_all["label"]

def make_dataset(dataset_name, data_dir, batch_size=128, sample_size=None, SOTA=False, normalize = True, classes_to_include = None):

    if dataset_name == 'cifar10':
        print('Dataset: CIFAR10.')
        if normalize:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        else:
            mean = [0, 0, 0]
            std = [1, 1, 1]
        if SOTA:
            trainset = CIFAR10(root=data_dir, train=True, download=True, transform=transforms.Compose([
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),

                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                ]))
        else:
            trainset = CIFAR10(root=data_dir, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                ]))

        testset = CIFAR10(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ]))
        num_classes = 10
    elif dataset_name == 'mnist':
        print('Dataset: MNIST.')
        if normalize:
            mean = (0.1307,)
            std = (0.3081,)
        else:
            mean = (0,)
            std = (1,)
        trainset = MNIST(root=data_dir, train=True, download=True, transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]))

        testset = MNIST(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]))
        num_classes = 10
    elif dataset_name == 'cifar10_random':
        print('Dataset: CIFAR10 with random label.')
        if normalize:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = (0,)
            std = (1,)
        trainset = CIFAR10RandomLabels(root=data_dir, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ]))

        testset = CIFAR10RandomLabels(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ]))
        num_classes = 10
    else:
        raise ValueError

    if sample_size is not None:
        total_sample_size = num_classes * sample_size
        cnt_dict = dict()
        total_cnt = 0
        indices = []
        for i in range(len(trainset)):

            if total_cnt == total_sample_size:
                break

            label = trainset[i][1]
            if label not in cnt_dict:
                cnt_dict[label] = 1
                total_cnt += 1
                indices.append(i)
            else:
                if cnt_dict[label] == sample_size:
                    continue
                else:
                    cnt_dict[label] += 1
                    total_cnt += 1
                    indices.append(i)

        train_indices = torch.tensor(indices)
        if classes_to_include != None:
            trainset = FilteredDataset(trainset, classes_to_include)
        # trainloader = DataLoader(
        #     trainset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), num_workers=1)
        subset_dataset = SubsetDataset(trainset, indices)
        trainloader = DataLoader(
            subset_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        print("len(trainloader)", len(trainloader))
        print("len(trainloader)", len(trainloader.dataset))
    else:
        if classes_to_include != None:
            trainset = FilteredDataset(trainset, classes_to_include)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)


    if classes_to_include != None:
        testset = FilteredDataset(testset, classes_to_include)
        num_classes = len(classes_to_include)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    return trainloader, testloader, num_classes


