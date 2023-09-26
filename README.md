# SegmentationUtils

Segmentation Utils is a repository that contains all code necessary to run instance segmentation analysis under the pyoseg package. It has 3 main functionalities:
1. Create the split annotations of the dataset into training, validation and testing given a fraction and option to be random or based on a clustering analysis. If based on a clustering analysis, it will fetch the same fraction of each cluster images to training, validation and testing.
2. Apply data augmentation to the dataset.
3. Create inference analysis, by comparing the predicted segmented objects versus the GT annotations, applying score filtering, size filtering and Non-Maximum Suppression. This inference analysis can also be done as a function of a clustering analysis, where each cluster will be analyzed separately.

## Installing pyoseg

To install the package, start by creating a conda environment:
```bash
conda create -n pyoseg python==3.11
```

And install the pyoseg package by using:
```bash
pip install pyoseg@git+https://github.com/FNLCR-DMAP/SegmentationUtils.git
```

Or clone the github repository, enter in the downloaded directory and install as:
```bash
pip install -e .
```

Some of the [unit tests](tests/) performed are using samples from [cocodataset](https://cocodataset.org/#download).

## Docker

If you prefer, there is a docker container with the environment already setup. Start by pulling the [image](https://github.com/users/hdegen/packages/container/package/segmentutils-ci):

```bash
docker pull ghcr.io/hdegen/segmentutils-ci:latest
```

And launch the application as:

```bash
docker run --rm --it ghcr.io/hdegen/segmentutils-ci
```

And once the container is up, grab all the repository scripts by running:

```bash
git clone git@github.com:FNLCR-DMAP/SegmentationUtils.git
cd SegmentationUtils
pip install -e .
```

## Running pyoseg

There are examples on how to run each of the functionalities inside the folder [examples](examples), basically showing how to:
1. Create a dataset split annotations: [here](examples/create_split.py)
2. Create an augmented dataset: [here](examples/data_augmentation.py)
3. Create the inference analysis: [here](examples/create_inference_plots.py)

More explanations below.

### 1 - Dataset split

Code examples [here](examples/create_split.py).

For creating the split annotations, there is essentially 3 ways of performing this actions:

#### i. By image ids

In this process, we only need the path to the coco annotations of the images, a list of the name of the images we want to create the annotations (without extension), and the name of the annotation output file:
```python
from pyoseg.split import create_annotations

annotations_path = "../data/coco_annotations/"
output_file = "single_image_annotation.json"
image_ids = ["P57_FOV14_0_1__1_1"]

create_annotations(annotations_path, output_file, image_ids)
```

#### ii. Randomly

For this process, we only need to insert the path for the coco annotations and the output path. The package will look into the folder, grab the annotation files, and split randomnly given the split fractions. As output, we will have 4 json files written at the output path: the split.json file which will have the name of the images and which split they were sent to, and the train_annotations.json, validation_annotations.json and test_annoations.json, which are each of the split annotations.
```python
from pyoseg.split import create_split

annotations_path = "../data/coco_annotations"
output_path = "random_split"

os.system(f"mkdir {output_path}")
config, train_ann, val_ann, test_ann = create_split(
    annotations_path=annotations_path,
    output_path=output_path,
    train_fraction=0.7,
    validation_fraction=0.2,
    test_fraction=0.1)
```

In case you have created an augmented dataset using pyoseg (next session), you can create an split based on these augmented annotations as well, for this purpose you will need to insert an additional information: the path where the augmented dataset is:
```python
from pyoseg.split import create_split

annotations_path = "../data/coco_annotations"
output_path = "random_split"
augmented_path = "../data/aug_data"  

os.system(f"mkdir {output_path}")
config, train_ann, val_ann, test_ann = create_split(
    annotations_path=annotations_path,
    output_path=output_path,
    train_fraction=0.7,
    validation_fraction=0.2,
    test_fraction=0.1,
    augmented_path=augmented_path)
```

#### iii. Based on a clustering file

If you have made an analysis to clusterize your images by similarity (so you can study afterwards which are the cluster of images that the model are performing better/worst), you can also made an split based on this clustering analysis. For this purpose, you will just need an additional information of where the clustering results in a csv format, where it is expected to have the name of the image as 1 column to identify the image and a second column as the name/number of the cluster that image belongs, such as:

```file
image,cluster
test1.png,1
test2.png,1
test3.png,2
```

There is an example file [here](tests/test_data/cluster.csv). The package at this point will separate the images per cluster and split each cluster of images on the given split fractions, ensuring all the clusters are represented in all the splits.

To use this functionality, you would run:
```python
from pyoseg.split import create_split_cluster

annotations_path = "../data/coco_annotations"
output_path = "cluster_split"
cluster_file = "../data/Membrane_small_img_clusters.csv"
cluster_column = 'PhenoGraph_clusters'
image_column = 'Images'
augmented_path = "augmented_dataset" # optional

os.system(f"mkdir {output_path}")
config, train_ann, val_ann, test_ann = create_split_cluster(
    cluster_file=cluster_file, cluster_column=cluster_column, image_column=image_column,
    output_path=output_path, annotations_path=annotations_path,
    train_fraction=0.7, validation_fraction=0.2, test_fraction=0.1,
    augmented_path=None)
# augmented path is optional, just in case we have an augmented dataset
```

### 2 - Augmentation

Code examples [here](examples/data_augmentation.py).

To create an augmented dataset, you will just need to insert the path to the coco annotations, a path to the images, an output path, the augmentation functions you would like to use and the number of augmented images your would like to create. In the example below, we create an augmented dataset 50x larger (50x augmentations per single image):

```python
from pyoseg.augmentation import augment_data

data_path = "../data/images/"
annotations_path = "../data/coco_annotations/"
output_path = "augmented_data"
number_augmentations = 50
augmentation_functions = [
    "RandomCrop", "HorizontalFlip", "VerticalFlip", "RandomRotate90", "GridDistortion", "Blur", "RandomBrightnessContrast", "RandomGamma"]
os.system(f"mkdir {output_path}")
augment_data(
    data_path=data_path,
    annotations_path=annotations_path,
    output_path=output_path,
    functions=augmentation_functions,
    times=number_augmentations)
```

For more advanced users the augmentation functions can be fully personalized. For this purpose, personalize your functions by modifying the augmentation dictionary:

```python
import pyoseg.augmentation.augmentations as augmented_dictionary
from albumentations import RandomCrop, HorizontalFlip, VerticalFlip

augmented_dictionary = {
    "RandomCrop": RandomCrop(p=1., height=256, width=256),
    "HorizontalFlip": HorizontalFlip(p=.5),
    "VerticalFlip": VerticalFlip(p=.5),
}
```

The code will be looking for this dictionary to perform the augmentations. You can personalize the probabilities of each augmentation function and their order, if needed.

### 3 - Instance Segmentation

Code examples [here](examples/create_inference_plots.py).

#### i. Instance Segmentation Analysis
To create the Instance Segmentation Analysis, you will have to have handy the GT annotation file created in earlier steps, and the prediction annotation file.

```python
from pyoseg.inference import create_inference_analysis

# The output folder:
output_path = "inference_analysis"
# Folder where the image data are located:
data_path = "../data/aug_data"
# File containing the GT annotations:
gt_annotations = "../data/aug_data_split/validation_annotations.json"
# File containing the inference annotations:
inf_annotations = "../../inference/coco_instances_results.json"

# Enable/disable information outputs
quiet_mode = True
# Filter detection objects smaller than <ize> pixels
size_filter = 25
# Filter detection objects with score below this threshold
score_filter = 0.45
# Non-Maximum Suppression threshold
nms_threshold = 0.3

os.system(f"mkdir {output_path}")
results = create_inference_analysis(
    data_path,
    gt_annotations,
    inf_annotations,
    output_path,
    size_filter=size_filter,
    score_filter=score_filter,
    nms_threshold=nms_threshold,
    quiet=quiet_mode)
```

This function will create several plots of precision, recall and F1 score per image (and characteristic) and evaluate this for all the images (not as an average, but by using the TP, FN and FP of all images to evaluate the metric).

In the output folder you will also find a folder called "NMS" where you can check all the objects that are being discarded by the Non-Maximum Supression algorithm and threshold that you are using.

You will also find a folder called "intersections", where you will find a figure per image that contains 3 plots, from left to right:
1. The image and its GT annotations
2. The image and the predicted annotations
3. The image and the highlighted False positive and False negative objects

You will also find at this output folder a file called "results.json" that have saved all the metrics and parameters obtained per image and overall.

#### ii. Instance Segmentation Cluster Analysis

After running the snipped code above, your can proceed and make an analysis per cluster using the "results" obtained above:

```python
from pyoseg.inference import inference_cluster_analysis

cluster_file = "../data/Membrane_small_img_clusters.csv"
cluster_column = 'PhenoGraph_clusters' 
image_column = 'Images'

inference_cluster_analysis(
    results, # results obtained above
    cluster_file=cluster_file,
    cluster_column=cluster_column,
    image_column=image_column,
    output_path=output_path)
```

This code will create a set of clustering plots, e.g., plots of precision, recall and F1 score per group of images based on the clustering file, where you can identify which set of images have better/worst performance and focus on them, by including more images of that cluster or augment this cluster more times.