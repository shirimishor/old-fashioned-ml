import albumentations as A
import cv2
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
import config.config
import random

# Define your augmentation pipeline
transform = A.Compose([
    A.Rotate(limit=40, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.5),
])

# Path to your dataset directory
dataset_dir = config.config.DATASET_DIR
output_dir = config.config.AUG_DIR
os.makedirs(output_dir, exist_ok=True)

# Step 1: Calculate the number of images per class
class_counts = {}
class_folders = {}

for class_folder in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_folder)
    if os.path.isdir(class_path):
        num_images = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
        class_counts[class_folder] = num_images
        class_folders[class_folder] = class_path

# Step 2: Determine the largest class size
max_images = max(class_counts.values())

# Step 3: Augment each class to match the largest class size
for class_name, current_images in class_counts.items():
    folder_path = class_folders[class_name]
    images_to_generate = max_images - current_images
    print(f"Generating {images_to_generate} images for class '{class_name}'")

    # Create output directory for each class
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    # Get list of images in the current class folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Generate the required number of augmented images
    for i in range(images_to_generate):
        # Randomly pick an image to augment
        img_name = random.choice(image_files)
        img_path = os.path.join(folder_path, img_name)
        
        # Load and apply augmentation
        img = cv2.imread(img_path)
        augmented = transform(image=img)
        
        # Save the augmented image
        output_img_path = os.path.join(class_output_dir, f"aug_{i}_{img_name}")
        cv2.imwrite(output_img_path, augmented["image"])

print("Augmentation complete.")
