import os
import sys

sys.path.append("../src")
from pyoseg.split import create_annotations, create_split, create_split_cluster


# Creating a single image dataset for overfitting
print("\n - Creating simple annotation")
# Path for the annotations in coco format:
annotations_path = "../data/coco_annotations/"
# Output file path:
output_file = "single_image_annotation.json"
# Array of image(s):
image_ids = ["P57_FOV14_0_1__1_1"]

# Call function
create_annotations(annotations_path, output_file, image_ids)


# Creating a randomized split based on the total number of files
print("\n - Creating randomized split")
# Path for the annotations in coco format:
annotations_path = "../data/coco_annotations"
# Path for output folder:
output_path = "random_split"

# Call function
os.system(f"mkdir {output_path}")
config, train_ann, val_ann, test_ann = create_split(
    annotations_path=annotations_path, output_path=output_path,
    train_fraction=0.7, validation_fraction=0.2, test_fraction=0.1,
    augmentation=None)


# Creating a randomized split based on a clustering analysis
print("\n - Creating randomized split based on clustering analysis")
# Path for the annotations in coco format:
annotations_path = "../data/coco_annotations"
# Path for output folder:
output_path = "cluster_split"
# csv file with name of images and clustering labels:
cluster_file = "../data/Membrane_small_img_clusters.csv"
# Column of the csv related to clustering labels:
cluster_column = 'PhenoGraph_clusters'
# Column of the csv related to the name of images:
image_column = 'Images'

# Call function
os.system(f"mkdir {output_path}")
config, train_ann, val_ann, test_ann = create_split_cluster(
    cluster_file=cluster_file, cluster_column=cluster_column,
    image_column=image_column, output_path=output_path,
    annotations_path=annotations_path, train_fraction=0.7,
    validation_fraction=0.2, test_fraction=0.1,
    augmentation=None)


# Example of augmentation configuration
augmentation = {
    "input_path": "../data/full_data",
    "train": {
        "functions": [
            "RandomCrop", "HorizontalFlip", "VerticalFlip",
            "RandomRotate90", "GridDistortion", "Blur",
            "RandomBrightnessContrast", "RandomGamma"],
        "times": 2},
    "val": {
        "functions": ["RandomCrop"],
        "times": 1},
    "test": {
        "functions": ["RandomCrop"],
        "times": 1}
}
