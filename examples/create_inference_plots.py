import os, sys
import matplotlib.pyplot as plt
import urllib.request as urllib

sys.path.append("../src")
from inference import create_inference_analysis

# Create inference plots and extract metric results
output_path = "output"                                                           # The output folder
data_path = "../data/full_data"                                                  # Folder where the image data are located
gt_annotations = "../../cluster/split_test/validation_annotations.json"          # File containing the GT annotations
inf_annotations = "../../6_full_200_split/inference/coco_instances_results.json" # File containing the inference annotations
quiet_mode = True                                                                # Enable/disable information outputs
size_filter = 25                                                                 # Filter detection objects smaller than <size> square pixels
os.system(f"mkdir {output_path}")

results = create_inference_analysis(data_path,gt_annotations,inf_annotations,output_path,size_filter=size_filter,quiet=quiet_mode)