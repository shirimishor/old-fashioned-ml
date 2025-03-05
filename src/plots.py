import matplotlib.pyplot as plt
import numpy as np
from dataset.pytorch_dataset import dataset, train_loader, train_set, test_set, mean, std, class_names


# Plotting finalized data

def denormalize(img, mean, std):
    # Since the images were normalized, they cannot be visualized properly 
    # In order to show an example batch, the images should be denormalized

    mean = mean.view(1, 1, 3)
    std = std.view(1, 1, 3)

    # Denormalize the image by reversing the normalization
    img = img * std + mean  
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
        denorm_img = denormalize(img.permute(1, 2, 0), mean, std).numpy()
        ax.imshow(np.clip(denorm_img, 0, 1))  # Clip to ensure values are within display range
        ax.set_title(class_names[example_labels[i].item()])
        ax.axis('off') 

 
    plt.tight_layout()
    plt.show()
