import os, sys

sys.path.append("../src")
from pyoseg.augmentation import augment_data


# Extend current dataset with data augmentation
print("\n - Augmenting data")
annotations_path = "../data/coco_annotations/"      # path for the annotations in coco format
data_path = "../data/full_data/"                    # path for the images folder
output_path = "augmented_dataset"                   # path for output folder
functions = [                                       # list of augmentation functions
    "RandomCrop", "HorizontalFlip", "VerticalFlip", "RandomRotate90", "GridDistortion",
    "Blur", "RandomBrightnessContrast", "RandomGamma"]

os.system(f"mkdir {output_path}")
augment_data(
    data_path=data_path,
    annotations_path=annotations_path,
    output_path=output_path,
    functions=functions,
    times=50)