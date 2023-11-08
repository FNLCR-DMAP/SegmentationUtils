import os
import sys

sys.path.append("../src")
from pyoseg.inference import create_inference_analysis, \
    inference_cluster_analysis

# Create inference plots and extract metric results
# The output folder:
output_path = "../../output_analysis"
# Folder where the image data are located:
data_path = "../data/full_data"
# File containing the GT annotations:
gt_annotations = "../data/annotations/validation_annotations.json"
# File containing the inference annotations:
inf_annotations = "../../data/inference/coco_instances_results.json"
# Enable/disable information outputs:
quiet_mode = True
# Filter detection objects smaller than <size> square pixels:
size_filter = 25
# Filter detection objects with score below this threshold:
score_filter = 0.45
# Non-Maximum Suppression threshold:
nms_threshold = 0.3

os.system(f"mkdir {output_path}")
results = create_inference_analysis(
    data_path, gt_annotations, inf_annotations, output_path,
    size_filter=size_filter, score_filter=score_filter,
    nms_threshold=nms_threshold, quiet=quiet_mode)

# Based on previous results, create an analysis per cluster
# csv file with name of images and clustering labels:
cluster_file = "../data/Membrane_small_img_clusters.csv"
# Column of csv related to clustering labels:
cluster_column = 'PhenoGraph_clusters'
# Column of csv related to the name of images:
image_column = 'Images'

inference_cluster_analysis(
    results, cluster_file=cluster_file, cluster_column=cluster_column,
    image_column=image_column, output_path=output_path)
