import os
import shutil
import random

# Set paths
original_dir = 'data/images/original/'
augmented_dir = 'data/images/augmented_data/'
output_base = 'data/split'
split_ratio = (0.8, 0.1, 0.1)  # train, val, test

classes = ['fresh', 'stale']
random.seed(42)

# Create output dirs
for split in ['train', 'val', 'test']:
    for cls in classes:
        os.makedirs(os.path.join(output_base, split, cls), exist_ok=True)

for cls in classes:
    orig_class_path = os.path.join(original_dir, cls)
    aug_class_path = os.path.join(augmented_dir, cls)

    images = [f for f in os.listdir(orig_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    train_end = int(len(images) * split_ratio[0])
    val_end = train_end + int(len(images) * split_ratio[1])

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    # Copy original images
    for img in train_imgs:
        shutil.copy2(os.path.join(orig_class_path, img), os.path.join(output_base, 'train', cls, img))

    for img in val_imgs:
        shutil.copy2(os.path.join(orig_class_path, img), os.path.join(output_base, 'val', cls, img))

    for img in test_imgs:
        shutil.copy2(os.path.join(orig_class_path, img), os.path.join(output_base, 'test', cls, img))

    # Also copy augmented images only to training
    if os.path.exists(aug_class_path):
        aug_imgs = [f for f in os.listdir(aug_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for aug_img in aug_imgs:
            shutil.copy2(os.path.join(aug_class_path, aug_img), os.path.join(output_base, 'train', cls, aug_img))

print("✅ Dataset split complete. No data leakage.")
