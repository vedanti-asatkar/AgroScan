import os
import numpy as np
from PIL import Image
from scipy import stats
import glob

def get_brightness(folder_path, max_images=200):
    brightness_values = []
    # Recursively find all image files, any depth
    image_paths = glob.glob(os.path.join(folder_path, '**', '*.jpg'), recursive=True)
    image_paths += glob.glob(os.path.join(folder_path, '**', '*.JPG'), recursive=True)
    image_paths += glob.glob(os.path.join(folder_path, '**', '*.png'), recursive=True)

    image_paths = image_paths[:max_images]

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("L")
            brightness = np.array(img).mean()
            brightness_values.append(brightness)
        except Exception:
            continue
    return np.array(brightness_values)
train_brightness = get_brightness('data/plantvillage')
incoming_brightness = get_brightness('data/plantdoc_flat')

print(f"Training set: {len(train_brightness)} images, mean brightness: {train_brightness.mean():.2f}")
print(f"Incoming set: {len(incoming_brightness)} images, mean brightness: {incoming_brightness.mean():.2f}")

ks_stat, p_value = stats.ks_2samp(train_brightness, incoming_brightness)

print(f"\nKS statistic: {ks_stat:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    print("DRIFT DETECTED — incoming data is statistically different from training data")
else:
    print("No significant drift detected")