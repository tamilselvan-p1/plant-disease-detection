import os
import shutil
import random

# Set source folder (where downloaded dataset folder lives) and target destination
SOURCE_DIR = "raw_plantvillage_dataset"  # Folder containing raw class subfolders
TARGET_DIR = "dataset"
TRAIN_RATIO = 0.8  # 80% training, 20% validation

classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

for class_name in classes:
    class_path = os.path.join(SOURCE_DIR, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)
    
    split_index = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_index]
    val_images = images[split_index:]
    
    # Create target folders
    train_class_dir = os.path.join(TARGET_DIR, "train", class_name)
    val_class_dir = os.path.join(TARGET_DIR, "val", class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    
    # Copy images
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_class_dir, img))

print("Dataset successfully structured into train/ and val/ folders!")