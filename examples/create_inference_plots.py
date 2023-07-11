import os, sys
import matplotlib.pyplot as plt
import urllib.request as urllib

sys.path.append("../src")
from pyoseg.inference import create_inference_analysis, inference_cluster_analysis

# Create inference plots and extract metric results
output_path = "output"                                                           # The output folder
data_path = "../data/full_data"                                                  # Folder where the image data are located
gt_annotations = "../../old/cluster/split_test/train_annotations.json"      # File containing the GT annotations
inf_annotations = "../../10_full_200000/inference_train/coco_instances_results.json" # File containing the inference annotations
quiet_mode = True                                                                # Enable/disable information outputs
size_filter = 25                                                                 # Filter detection objects smaller than <size> square pixels
score_filter = 0.35#35                                                              # Filter detection objects with score below this threshold
nms_threshold = 0.2                                                              # Non-Maximum Suppression threshold
os.system(f"mkdir {output_path}")

results = create_inference_analysis(
    data_path,gt_annotations,inf_annotations,output_path,
    size_filter=size_filter,score_filter=score_filter,
    nms_threshold=nms_threshold,quiet=quiet_mode)

# Based on previous results, create an analysis per cluster
cluster_file = "../data/Membrane_small_img_clusters.csv"                         # csv file with name of images and clustering labels
cluster_column = 'PhenoGraph_clusters'                                           # column of csv related to clustering labels
image_column = 'Images'                                                          # column of csv related to the name of images
inference_cluster_analysis(
    results,cluster_file=cluster_file,cluster_column=cluster_column,
    image_column=image_column,output_path=output_path)