import matplotlib.pyplot as plt
import numpy as np
import random

def imshow(img, ax, source="cifar10"):
    """ Function to show a torch image on a given Matplotlib axis. """
    img = img.numpy().transpose((1, 2, 0))  # Convert from PyTorch to numpy format
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

def visualize_image(image, label, source="cifar10"):
    fig, ax = plt.subplots()  # Create a new figure and an axes.
    ax.set_title(f'Label: {label}')
    imshow(image, ax, source=source)  # Pass ax to imshow
    plt.show()
    

def visualize_autoencoder_results(original_images, reconstructed_images, num_images=5):
    fig, axes = plt.subplots(2, num_images, figsize=(10, 2))  # Create a subplot with 2 rows and num_images columns
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

def visualize_image_loader(dataloader, index=0, source="cifar10"):
    dataset = dataloader.dataset
    if index >= len(dataset) or index < 0:
        print(f"\nIndex out of bound, please pick another index in range (0, {len(dataset)-1})")
        return

    image, label = dataset[index]  # This returns a tuple (image, label)

    fig, ax = plt.subplots()  # Create a new figure and an axes.
    ax.set_title(f'Label: {label}')
    imshow(image, ax, source=source)  # Pass ax to imshow
    plt.show()

def visualize_images_loader(dataloader, num_classes=10, seed=512, source="cifar10"):
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
