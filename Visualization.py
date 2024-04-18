import torchvision
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    """ Function to show an image. """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def visualize_one_image_per_class(dataloader):
    # Dictionary to hold one image per class
    class_images = {}
    class_labels = {}
    
    # Iterate over the dataloader to collect one image per class
    for images, labels in dataloader:
        for image, label in zip(images, labels):
            label = label.item()
            if label not in class_images:
                # Convert tensor image to numpy for visualization
                class_images[label] = image.numpy().transpose(1, 2, 0)
                class_labels[label] = label
            # Stop iterating once we have one image per class
            if len(class_images) == len(dataloader.dataset.classes):
                break
        if len(class_images) == len(dataloader.dataset.classes):
            break
    
    # Number of classes
    num_classes = len(class_images)
    
    # Create a plot to display the images
    fig, axs = plt.subplots(1, num_classes, figsize=(num_classes * 3, 3))
    fig.subplots_adjust(top=0.8)
    
    # Check if we are handling a single subplot or multiple
    if num_classes == 1:
        axs = [axs]  # Make it iterable
    
    # Display each image and label
    for ax, (label, image) in zip(axs, class_images.items()):
        ax.imshow(image)
        ax.set_title(f'Class {label}')
        ax.axis('off')
    
    plt.show()
