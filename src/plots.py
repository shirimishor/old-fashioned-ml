import matplotlib.pyplot as plt
import numpy as np

#import os
#import sys
#sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))

from dataset.pytorch_dataset import dataset, train_loader, train_set, test_set, mean, std, class_names


def denormalize(img, mean, std):
    # Since the images were normalized, they cannot be visualized properly 
    # In order to show an example batch, the images should be denormalized

    # Ensure mean and std have the correct shape for broadcasting
    mean = mean.view(1, 1, 3)  # Reshape mean to (1, 1, 3)
    std = std.view(1, 1, 3)    # Reshape std to (1, 1, 3)

    # Denormalize the image by reversing the normalization
    img = img * std + mean  # Rescale back to original range
    return img



if __name__ == "__main__":
    # Print dataset labels

    print(f"Total dataset size: {len(dataset)}")
    print(f"Train set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")


    print("Classes:", class_names)

    # Visualize an example batch
    example_imgs, example_labels = next(iter(train_loader))

    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(16, 8))
    axes = axes.flatten()

    # Plot each image in the grid
    for i, img in enumerate(example_imgs):
        ax = axes[i]
        # Denormalize image (make sure to permute the image for correct display)
        denorm_img = denormalize(img.permute(1, 2, 0), mean, std).numpy()  # Convert tensor to numpy
        ax.imshow(np.clip(denorm_img, 0, 1))  # Clip to ensure values are within display range
        ax.set_title(class_names[example_labels[i].item()])
        ax.axis('off')  # Hide the axis

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
