import os, sys
sys.path.append("../src")
from pyoseg.split import create_annotations, create_split, create_split_cluster


# Creating a single image dataset for overfitting:
print(f"\n - Creating simple annotation")
annotations_path = "../data/coco_annotations/"           # path for the annotations in coco format
output_file = "single_image_annotation.json"             # output file path
image_ids = ["P57_FOV14_0_1__1_1"]                       # array of image(s)
create_annotations(annotations_path,output_file,image_ids)


# Creating a randomized split based on the total number of files
print(f"\n - Creating randomized split")
annotations_path = "../data/coco_annotations"            # path for the annotations in coco format (must be the original data without augmentation)
output_path = "random_split"                             # path for output folder

os.system(f"mkdir {output_path}")
config, train_ann, val_ann, test_ann = create_split(
    annotations_path=annotations_path, output_path=output_path,
    train_fraction=0.7, validation_fraction=0.2, test_fraction=0.1,
    augmentation=None) # augmentation path is optional


# Creating a randomized split based on a clustering analysis
print(f"\n - Creating randomized split based on clustering analysis")
annotations_path = "../data/coco_annotations"            # path for the annotations in coco format
output_path = "cluster_split"                            # path for output folder
cluster_file = "../data/Membrane_small_img_clusters.csv" # csv file with name of images and clustering labels
cluster_column = 'PhenoGraph_clusters'                   # column of csv related to clustering labels
image_column = 'Images'                                  # column of csv related to the name of images

os.system(f"mkdir {output_path}")
config, train_ann, val_ann, test_ann = create_split_cluster(
    cluster_file=cluster_file, cluster_column=cluster_column, image_column=image_column,
    output_path=output_path, annotations_path=annotations_path,
    train_fraction=0.7, validation_fraction=0.2, test_fraction=0.1,
    augmentation=None)


# Example of augmentation configuration
augmentation = {
    "input_path": "../data/full_data",
    "train": {
        "functions": [
            "RandomCrop", "HorizontalFlip", "VerticalFlip", "RandomRotate90",
            "GridDistortion", "Blur", "RandomBrightnessContrast", "RandomGamma"],
        "times": 2},
    "val": {
        "functions": ["RandomCrop"],
        "times": 1},
    "test": {
        "functions": ["RandomCrop"],
        "times": 1}
}

