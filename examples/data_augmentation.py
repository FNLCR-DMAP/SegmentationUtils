import os, sys

sys.path.append("../src")
from pyoseg.augmentation import augment_data, random_crop


# Extend current dataset with data augmentation
print(f"\n - Augmenting data")
annotations_path = "../data/coco_annotations/"           # path for the annotations in coco format
data_path = "../data/full_data/"                         # path for the images folder
output_path = "test_augmentation_crop"                        # path for output folder
os.system(f"mkdir {output_path}")
#random_crop(data_path=data_path, annotations_path=annotations_path, output_path=output_path, size=(256,256,10))

# Extend current dataset with data augmentation
print(f"\n - Augmenting data")
annotations_path = "../data/coco_annotations/"           # path for the annotations in coco format
data_path = "../data/full_data/"                         # path for the images folder
output_path = "test_augmentation"                        # path for output folder
os.system(f"mkdir {output_path}")
augment_data(data_path=data_path, annotations_path=annotations_path, output_path=output_path)
sys.exit()
