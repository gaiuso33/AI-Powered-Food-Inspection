from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.preprocessing import image
from PIL import UnidentifiedImageError
import numpy as np

# Set paths
input_folder = 'data/'  # Your original 200 images folder
output_folder = 'augmented_data/'
os.makedirs(output_folder, exist_ok=True)

# Valid image extensions
image_extensions = ('.jpg', '.jpeg', '.png', 'heif')

# Create augmentation settings
augmentor = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.5, 1.5],
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Class folders (must exist inside `data/`)
classes = ['fresh', 'stale']

# Create output folders
for cls in classes:
    os.makedirs(os.path.join(output_folder, cls), exist_ok=True)

# Apply augmentation
for cls in classes:
    path = os.path.join(input_folder, cls)
    save_path = os.path.join(output_folder, cls)

    for img_name in os.listdir(path):
        if not img_name.lower().endswith(image_extensions):
            print(f"Skipping non-image file: {img_name}")
            continue

        img_path = os.path.join(path, img_name)
        try:
            img = image.load_img(img_path, target_size=(224, 224))
        except UnidentifiedImageError:
            print(f"Skipped corrupt image: {img_name}")
            continue

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Generate 10 augmented images per original
        i = 0
        for batch in augmentor.flow(x, batch_size=1, save_to_dir=save_path, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= 10:
                break

        print(f"Augmented: {img_name} → {cls} ({i} new images)")
