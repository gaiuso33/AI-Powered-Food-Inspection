from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
import pillow_heif
import os
import numpy as np

# Register HEIC support with Pillow
pillow_heif.register_heif_opener()

# Paths
input_folder = 'data/images/original/' 
output_folder = 'data/images/augmented_data/'
os.makedirs(output_folder, exist_ok=True)

# Image extensions to process (after conversion)
image_extensions = ('.jpg', '.jpeg', '.png')

# Define classes
classes = ['fresh', 'stale']

# Create class folders in output
for cls in classes:
    os.makedirs(os.path.join(output_folder, cls), exist_ok=True)

# Image augmentation settings
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

# Loop through each class folder
for cls in classes:
    class_input_path = os.path.join(input_folder, cls)
    class_output_path = os.path.join(output_folder, cls)

    for img_name in os.listdir(class_input_path):
        img_path = os.path.join(class_input_path, img_name)

        # Convert HEIC to JPG
        if img_name.lower().endswith('.heic'):
            jpg_name = os.path.splitext(img_name)[0] + '.jpg'
            jpg_path = os.path.join(class_input_path, jpg_name)
            try:
                heif_image = Image.open(img_path)
                heif_image.save(jpg_path, format="JPEG")
                print(f"Converted HEIC → JPG: {img_name} → {jpg_name}")
            except Exception as e:
                print(f"Failed to convert {img_name}: {e}")
            continue  # Skip the original HEIC file

        # Skip unsupported formats
        if not img_name.lower().endswith(image_extensions):
            print(f"Skipping unsupported file: {img_name}")
            continue

        # Load and augment
        try:
            img = image.load_img(img_path, target_size=(224, 224))
        except (UnidentifiedImageError, OSError) as e:
            print(f"Corrupt/unreadable image skipped: {img_name} | {e}")
            continue

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Generate and save 10 augmented images
        for i, batch in enumerate(augmentor.flow(x, batch_size=1, save_to_dir=class_output_path, save_prefix='aug', save_format='jpeg')):
            if i >= 10:
                break

        print(f"Augmented: {img_name} → {cls} (10 images)")
