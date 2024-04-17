import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    """ Function to show an image. """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def visualize_dataset(dataset_name='CIFAR10', augment=False):
    """ Visualize images from CIFAR10 or MNIST dataset. """
    data_loader = load_data(dataset_name, augment)
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    # Show images
    imshow(torchvision.utils.make_grid(images))
    # Print labels
    print('Labels:', labels)
