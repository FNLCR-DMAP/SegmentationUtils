import os
import sys

sys.path.append("../src")
from pyoseg.augmentation import augment_data

print("\n - Augmenting data")
# Extend current dataset with data augmentation
# Path for the annotations in coco format:
annotations_path = "../data/coco_annotations/"
# Path for the images folder:
data_path = "../data/full_data/"
# Path for output folder:
output_path = "augmented_dataset"
# list of augmentation functions:
functions = [
    "RandomCrop", "HorizontalFlip", "VerticalFlip", "RandomRotate90",
    "GridDistortion", "Blur", "RandomBrightnessContrast", "RandomGamma"]

os.system(f"mkdir {output_path}")
augment_data(
    data_path=data_path,
    annotations_path=annotations_path,
    output_path=output_path,
    functions=functions,
    times=50)
