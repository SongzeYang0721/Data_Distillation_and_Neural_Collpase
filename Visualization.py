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

def visualize_images(images, labels, source="cifar10"):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))  # Create a row of subplots

    # In case there's only one image, `axes` will not be an array but a single object
    if num_images == 1:
        axes = [axes]

    for ax, img, lbl in zip(axes, images, labels):
        imshow(img, ax, source=source)  # Pass ax to imshow
        ax.set_title(f'Label: {lbl}')
        ax.axis('off')  # Hide axis

    plt.tight_layout()  # Adjust subplots to fit in the figure area
    plt.show()

def visualize_image_loader(dataloader, index=0, source="cifar10"):
    dataset = dataloader.dataset
    if index >= len(dataset) or index < 0:
        print(f"\nIndex out of bound, please pick another index in range (0, {len(dataset)-1})")
        return

    image, label = dataset[index]  # This returns a tuple (image, label)

    fig, ax = plt.subplots()  # Create a new figure and an axes.
    ax.set_title(f'Label: {label}')
    # ax.set_title("Original")
    imshow(image, ax, source=source)  # Pass ax to imshow
    plt.show()

def visualize_images_per_class_loader(dataloader, num_classes=10, seed=512, source="cifar10"):
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
        imshow(img, ax, source=source)
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