import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sophia import SophiaG
import torch.optim.lr_scheduler as lrs


def set_seed(manualSeed=666):
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(manualSeed)


def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    
    if args.sep_decay:
        wd_term = 0
    else:
        wd_term = args.weight_decay

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9,
                  'lr': args.lr,
                  'weight_decay': wd_term#args.weight_decay
        }
    elif args.optimizer == 'Adam':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'lr': args.lr,
            'weight_decay': wd_term#args.weight_decay
        }
    elif args.optimizer == 'LBFGS':
        optimizer_function = optim.LBFGS
        kwargs = {'lr': args.lr,
                  'history_size': args.history_size,
                  'line_search_fn': 'strong_wolfe'
        }
    elif args.optimizer == 'AdamW':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'lr': args.lr,
            'weight_decay': wd_term#args.weight_decay
        }
    elif args.optimizer == 'Adadelta':
        optimizer_function = optim.Adadelta
        kwargs = {
            'lr': args.lr,
            'weight_decay': wd_term#args.weight_decay
        }
    elif args.optimizer == "Sophia":
        optimizer_function = SophiaG
        kwargs = {
            "lr": args.lr
        }

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.patience,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )
    elif args.decay_type == 'cosine':
        scheduler = lrs.CosineAnnealingLR(
            my_optimizer,
            T_max=args.epochs,  # Number of epochs until the first restart
        )

    return scheduler


def make_criterion(args):
    if args.loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'MSE':
        criterion = nn.MSELoss()
    return criterion


def make_criterion_AE(args):
    if args.loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(reduction='sum')
    elif args.loss == 'MSE':
        criterion = nn.MSELoss(reduction='sum')
    return criterion


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters])


def print_and_save(text_str, file_stream):
    print(text_str)
    print(text_str, file=file_stream)


def compute_accuracy(output, target, topk=(1,), is_distance=False):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    if is_distance:
        # For distance matrices, smallest distances imply nearest neighbors.
        # Invert the selection logic by choosing smallest values.
        _, pred = output.topk(maxk, dim=1, largest=False, sorted=True)
    else:
        # For prediction scores, largest values imply highest confidence predictions.
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


### Visualization Task

import matplotlib.pyplot as plt
import numpy as np
import random
import torch

def imshow(img, ax, normalize_data = False, source="cifar10"):
    """ Function to show a torch image on a given Matplotlib axis. """
    img = img.numpy().transpose((1, 2, 0))  # Convert from PyTorch to numpy format
    if normalize_data:
        if source == "cifar10":
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
        elif source == "mnist":
            mean = np.array([0.1307])
            std = np.array([0.3081])
        else:
            raise ValueError("Unsupported dataset specified")

        img = img * std + mean  # Denormalize
    img = np.clip(img, 0, 1)  # Clip values to ensure they're between 0 and 1
    ax.imshow(img)
    ax.axis('off')  # Hide axis

def random_sample_images_index(dataloader,num_images = 10,seed = 512):
    dataset = dataloader.dataset
    random.seed(seed)
    index = random.sample(range(0,len(dataset)-1), num_images)
    return index

def images_from_index(dataset, indices):
    images = []
    labels = []
    for idx in indices:
        image, label = dataset[idx]
        images.append(image)
        labels.append(label)
    
    # Stack all images and labels into single tensors
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    return images, labels

def random_sample_images(dataloader, num_images=10, seed=512):
    random.seed(seed)
    dataset = dataloader.dataset
    indices = random.sample(range(len(dataset)), num_images)
    return images_from_index(dataset, indices)

def visualize_images(images, labels, normalize_data = False, source="cifar10"):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))  # Create a row of subplots

    # In case there's only one image, `axes` will not be an array but a single object
    if num_images == 1:
        axes = [axes]

    for ax, img, lbl in zip(axes, images, labels):
        imshow(img, ax, normalize_data, source=source)  # Pass ax to imshow
        ax.set_title(f'Label: {lbl}')
        ax.axis('off')  # Hide axis

    plt.tight_layout()  # Adjust subplots to fit in the figure area
    plt.show()

def visualize_image_loader(dataloader, index=0, normalize_data = False, source="cifar10"):
    dataset = dataloader.dataset
    if index >= len(dataset) or index < 0:
        print(f"\nIndex out of bound, please pick another index in range (0, {len(dataset)-1})")
        return

    image, label = dataset[index]  # This returns a tuple (image, label)

    fig, ax = plt.subplots()  # Create a new figure and an axes.
    ax.set_title(f'Label: {label}')
    # ax.set_title("Original")
    imshow(image, ax, normalize_data, source=source)  # Pass ax to imshow
    plt.show()

def visualize_images_per_class_loader(dataloader, num_classes=10, seed=512, normalize_data = False, source="cifar10"):
    dataset = dataloader.dataset
    class_indices = {i: [] for i in range(num_classes)}

    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    random.seed(seed)
    selected_indices = [random.choice(class_indices[i]) for i in range(num_classes)]
    selected_images = [dataset[i][0] for i in selected_indices]
    selected_labels = [dataset[i][1] for i in selected_indices]

    ranked_labels, ranked_images = zip(*sorted(zip(selected_labels, selected_images), key=lambda x: x[0]))

    fig, axes = plt.subplots(1, num_classes, figsize=(num_classes * 2, 3))
    for ax, img, label in zip(axes, ranked_images, ranked_labels):
        imshow(img, ax, normalize_data, source=source)
        ax.set_title(f'Label: {label}')

    plt.tight_layout()
    plt.show()

def visualize_autoencoder_results(dataloader, device, num_images=5, seed = 512):
    fig, axes = plt.subplots(2, num_images, figsize=(10, 2))  # Create a subplot with 2 rows and num_images columns
    dataset = dataloader.dataset
    random.seed(seed)
    index = random.sample(range(0,len(dataset)-1), num_images)
    
    inputs = torch.stack([dataset[i][0] for i in index]).to(device)
    for i in range(num_images):
        # Display original images
        ax = axes[0, i]
        img = original_images[i].detach().cpu().numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)  # Ensure the pixel values are valid
        ax.imshow(img)
        ax.set_title("Original")
        ax.axis('off')

        # Display reconstructed images
        ax = axes[1, i]
        img = reconstructed_images[i].detach().cpu().numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)  # Ensure the pixel values are valid
        ax.imshow(img)
        ax.set_title("Reconstructed")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()