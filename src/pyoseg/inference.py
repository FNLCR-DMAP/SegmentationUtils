import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import urllib.request as urllib
import pycocotools.mask as mask
import seaborn as sns
import json
import cv2
import tifffile
import skimage
from shapely.geometry import Polygon
from skimage.color import rgb2gray
from pyoseg.split import get_split_ids
from PIL import Image

def get_iou(inference, gt, quiet=True):
    """
    Calculates the Intersection over Union (IoU) between two binary masks.
    
    Parameters:
        inference (ndarray): The predicted binary mask.
        gt (ndarray): The ground truth binary mask.
        quiet (bool, optional): Whether or not to print the number of nuclei in the ground truth and inference masks. Defaults to True.
    
    Returns:
        ndarray: The IoU between the two masks.
    """
    true_objects = np.unique(gt)
    pred_objects = np.unique(inference)
    if not quiet:
        print("ground truth nuclei:", len(true_objects)-1)
        print("Inference nuclei:", len(pred_objects)-1)
    true_bins = np.append(true_objects, true_objects[-1] + 1)
    pred_bins = np.append(pred_objects, pred_objects[-1] + 1)
    
    intersection, xedges, yedges = np.histogram2d(gt.flatten(), inference.flatten(), bins=(true_bins, pred_bins))
    area_true = np.histogram(gt, bins = true_bins)[0]
    area_pred = np.histogram(inference, bins = pred_bins)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    union = area_true + area_pred - intersection
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9
    iou = intersection / union
    return iou


def precision_at(threshold,iou,quiet=True):
    """
    Calculates the precision at a given threshold for a set of intersection over union (IOU) values.
    
    Args:
        threshold (float): The IOU threshold to calculate precision at.
        iou (ndarray): The array of IOU values.
        quiet (bool, optional): Whether to suppress printing intermediate results. Defaults to True.
    
    Returns:
        tuple: A tuple containing the number of true positives, false positives, and false negatives.
    """
    matches = iou > threshold
    true_positives =  np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    
    matches_01 = iou > 0.1
    merges = np.sum(np.sum(matches_01, axis = 0) > 1)
    splits = np.sum(np.sum(matches_01, axis = 1) > 1)
    if not quiet:
        print("merges:{0}, splits:{1}, true_positives:{2} false_positives:{3}, false_negatives:{4}".format(merges, splits, tp, fp, fn))
    return true_positives, false_positives, false_negatives


def get_permuted_lookup(ids):
    """
    Permute the instance ids for better display.

    Parameters:
        ids (list): A list of instance ids.

    Returns:
        numpy.ndarray: A permuted lookup array.
    """
    # Permute the instance ids for better display
    np.random.seed(2)
    max_id = np.max(ids)
    max_id = 2000
    lookup = np.random.permutation(max_id + 2)
    # Make sure background stays as background
    lookup = lookup[lookup != 0]
    lookup[0] = 0
    return lookup


def get_intersections(gt,pred):
    """
    Calculate the intersections between two sets of nuclei masks.

    Parameters:
        gt (ndarray): Ground truth nuclei masks.
        pred (ndarray): Predicted nuclei masks.

    Returns:
        list: A list containing two elements:
            - inference_permuted (ndarray): Permuted predicted nuclei masks.
            - gt_xor_inference (ndarray): Ground truth XOR predicted nuclei masks with highlighted false negatives and false positives.
    """
    iou = get_iou(pred,gt,1)
    tp, fp, fn = precision_at(0.7,iou,1)
    fn_indexes = np.nonzero(fn)[0] + 1
    fp_indexes = np.nonzero(fp)[0] + 1

    # Highlight false negative IDs from ground truth
    gt_fn = np.zeros(gt.shape)
    for nuclei_id in fn_indexes:
        gt_fn[gt == nuclei_id] = nuclei_id

    gt_xor_inference = np.bitwise_xor(gt != 0, pred != 0) * 2500
    lookup_gt = get_permuted_lookup(gt)
    gt_fn_permuted = lookup_gt[gt_fn.astype("uint16")]
    
    # Highlight all the false positive nuclei in addition to xor
    np.copyto(gt_xor_inference, gt_fn_permuted, where = gt_fn != 0)
        
    lookup_inference = get_permuted_lookup(pred.astype("uint16"))
    inference_permuted = lookup_inference[pred.astype("uint16")]
    return [inference_permuted, gt_xor_inference]


def plot_gt_image(img,gt,name='gt.png'):
    """
    Plot a ground truth image with overlays on top of the original image.

    Parameters:
        img (ndarray): The original image.
        gt (ndarray): The ground truth mask.
        name (str, optional): The name of the output image file. Defaults to 'gt.png'.

    Raises:
        AssertionError: If the shape of the image and the ground truth mask are different.

    Returns:
        None
    """
    image = rgb2gray(img)
    assert image.shape == gt.shape, "Image and masks have different sizes!"
        
    fig, ax = plt.subplots(1,1, figsize=(32,32))
    fig.tight_layout(pad=-2.6)
    nuclei_cmap = "gist_ncar"
    inf_alpha = 0.1
        
    # Cconvert zero to black color in gitst_ncar
    if nuclei_cmap == "gist_ncar":
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        gist_ncar = cm.get_cmap('gist_ncar', 256)
        newcolors = gist_ncar(np.linspace(0, 1, 256))
        black = np.array([0, 0, 0, 1])
        magneta = np.array([1, 0, 1, 1])
        newcolors[0, :] = black
        newcolors[255, :] = magneta
        newcmp = ListedColormap(newcolors)
        nuclei_cmap = newcmp
        
        ax.imshow(image, cmap=plt.cm.gray)
        ax.imshow(gt, cmap=nuclei_cmap, alpha=inf_alpha)

        # Turn off axis and y axis
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        fig.savefig(name, bbox_inches = 'tight',pad_inches = 0)
        plt.close()


def plot_intersections(gt,gt_per,gt_xor,image,name='intersections.png'):
    """
        Plot the intersections between ground truth, ground truth with perturbations, and ground truth with XOR operation.
        
        Parameters:
            gt: Ground truth image
            gt_per: Ground truth image with perturbations
            gt_xor: Ground truth image with XOR operation
            image: Input image
            name: Name of the output image file (default: "intersections.png")
        
        Returns:
            None
    """
    nfigures = 3
    dim = (0,200,0,200)

    image = rgb2gray(image)
        
    fig, axes = plt.subplots(1,nfigures, figsize=(32,32))
    fig.tight_layout(pad=-2.6)
    nuclei_cmap = "gist_ncar"
    inf_alpha = 0.3
    err_alpha = 0.25
    mag = 1
        
    # Cconvert zero to black color in gitst_ncar
    if nuclei_cmap == "gist_ncar":
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        gist_ncar = cm.get_cmap('gist_ncar', 256)
        newcolors = gist_ncar(np.linspace(0, 1, 256))
        black = np.array([0, 0, 0, 1])
        magneta = np.array([1, 0, 1, 1])
        newcolors[0, :] = black
        newcolors[255, :] = magneta
        newcmp = ListedColormap(newcolors)
        nuclei_cmap = newcmp
        
        axes_id = -1

        axes_id += 1
        axes[axes_id].imshow(image, cmap=plt.cm.gray)
        axes[axes_id].imshow(gt, cmap=nuclei_cmap, alpha=inf_alpha)

        axes_id += 1
        axes[axes_id].imshow(image, cmap=plt.cm.gray)
        axes[axes_id].imshow(gt_per, cmap=nuclei_cmap, alpha=inf_alpha)

        axes_id += 1
        axes[axes_id].imshow(image, cmap=plt.cm.gray)
        axes[axes_id].imshow(gt_xor, cmap=nuclei_cmap, alpha=err_alpha)

        x_loc = int((dim[1] - dim[0]) * 3 / 4)
        y_loc = int((dim[3] - dim[2]) * 7 / 8)
        
        lower_write = dim[1] - dim[0] - 20 / (500 / (dim[1] - dim[0]))
        rect = patches.Rectangle((lower_write - mag, y_loc), mag, 8, color ='w')
        axes[axes_id].add_patch(rect)

        # Turn off axis and y axis
        for axes_id in range(0, nfigures):
            axes[axes_id].get_xaxis().set_visible(False)
            axes[axes_id].get_yaxis().set_visible(False)

        fig.savefig(name, bbox_inches = 'tight',pad_inches = 0)
        plt.close()


def get_prediction_annotations(file):
    """
    Reads a JSON file and returns a dictionary containing predictions indexed by image IDs.

    Parameters:
        file: A string representing the path to the JSON file.

    Returns:
        pred: A dictionary where the keys are image IDs (as strings) and the values are lists of predictions.
    """
    with open(file,'r') as f:
        p = json.load(f)
    
    pred = {}
    for i in p:
        if str(i['image_id']) not in pred.keys():
            pred[str(i['image_id'])] = []
        pred[str(i['image_id'])].append(i)
    
    return pred


def get_gt_annotations(file,ids):
    """
    Reads a JSON file and extracts the ground truth (gt) annotations for the given ids.

    Parameters:
        file (str): The path to the JSON file.
        ids (list): A list of ids for which to extract the annotations.

    Returns:
        dict: A dictionary containing the ground truth annotations for each id.
    """
    with open(file,'r') as f:
        d = json.load(f)
    
    gt = {}
    for i in ids:
        gt[str(i)] = {}
        gt[str(i)]['images'] = []
        gt[str(i)]['annotations'] = []
        for img in d['images']:
            if img['id'] == i:
                gt[str(i)]['images'].append(img)
        for ann in d['annotations']:
            if ann['image_id'] == i:
                gt[str(i)]['annotations'].append(ann)

    return gt


def annotation_poly_to_tiff(json_data,height,width,output_name="",ann_type="All"):
    """
    Converts a polygon annotation in a JSON file to a TIFF image.

    Parameters:
        json_data (dict): A dictionary containing the JSON data.
        height (int): The height of the TIFF image.
        width (int): The width of the TIFF image.
        output_name (str, optional): The name of the output TIFF file. Defaults to "".
        ann_type (str, optional): The type of annotation to convert. Defaults to "All".

    Returns:
        numpy.ndarray: An array representing the converted annotations as a TIFF image.
    """
    objects = json_data["annotations"]
    masks = np.zeros((height, width), dtype=np.uint16)

    #For every object
    object_id = 1
    for obj_dict in objects:
        if ann_type == "All" or obj_dict["name"] == ann_type: # if converting membrane annotations to png/tif, replace "Nucleus" with "Membrane"
            poly_list = obj_dict["segmentation"][0]
            contour_list = [[poly_list[i*2], poly_list[i*2+1]] for i in range(int(len(poly_list)/2))]
            nd_contour = np.array(contour_list).astype("int64")
            cv2.fillPoly(masks,pts=[nd_contour], color=object_id)

            object_id += 1 
    
    if output_name != "":
        tifffile.imwrite(output_name, masks)
    return masks


def polygon_from_mask(maskedArr): # https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    """
    Given a masked array, this function finds the contours of the mask and returns the polygon, bounding rectangle, and area.

    Parameters:
        maskedArr (ndarray): The masked array representing the image mask.

    Returns:
        tuple: A tuple containing the polygon coordinates, bounding rectangle coordinates, and area of the mask.

    Notes:
        - The function assumes that the maskedArr is a binary mask where the foreground is represented by non-zero values.
        - The function only considers polygons with at least 3 points (6 coordinates).
        - If no valid polygons are found, the function returns [None].
    """
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    if len(segmentation) == 0:
        return [None]
        
    RLEs = mask.frPyObjects(segmentation, maskedArr.shape[0], maskedArr.shape[1])
    RLE = mask.merge(RLEs)
    area = mask.area(RLE)
    [x, y, w, h] = cv2.boundingRect(maskedArr)

    return segmentation[0], [x, y, w, h], area


def sort_objects_by(contours,parameter,ascending=True):
    """
    Sorts a list of objects by a given parameter in ascending or descending order.

    Parameters:
        contours (list): A list of objects to be sorted.
        parameter (list): A list of parameters to sort the objects by.
        ascending (bool, optional): Sort the objects in ascending order if True, descending order if False. Defaults to True.

    Returns:
        tuple: A tuple containing the sorted list of objects and the sorted list of parameters.
    """
    sorted_contours = contours.copy()
    sorted_parameter = parameter.copy()
    for i in range(len(sorted_parameter)):
        min_max = np.min(sorted_parameter[i:]) if ascending else np.max(sorted_parameter[i:])
        for j in range(i,len(sorted_parameter)):
            if sorted_parameter[j] == min_max:
                val = sorted_parameter[i]
                sorted_parameter[i] = sorted_parameter[j]
                sorted_parameter[j] = val
                cont = sorted_contours[i].copy()
                sorted_contours[i] = sorted_contours[j].copy()
                sorted_contours[j] = cont.copy()
                break
    return sorted_contours, sorted_parameter


def non_maximum_suppression(contours,scores,height,width,threshold=0.3,output_name="",quiet=True):
    """
    Performs non-maximum suppression on a list of contours and scores.
    
    Parameters:
        contours: A list of contours.
        scores: A list of scores corresponding to each contour.
        height: The height of the image.
        width: The width of the image.
        threshold: The IOU threshold for suppression (default: 0.3).
        output_name: The name of the output file (default: "").
        quiet: Whether to suppress intermediate print statements (default: True).
    
    Returns:
        suppressed_contours: A list of contours after suppression.
        suppressed_scores: A list of scores after suppression.
    """
    if not quiet:
        print(f"NMS IOU threshold: {threshold}")
    sorted_contours, sorted_scores = sort_objects_by(contours,scores,ascending=False)
    n_objects = len(sorted_contours)
    suppressed_contours, suppressed_scores, suppressed_indexes = [],[],[]
    
    nms_images = np.zeros((height, width), dtype=np.uint16)
    for i in range(n_objects):
        obs_color = 1
        if i in suppressed_indexes:
            continue
        contour_i = sorted_contours[i]
        poly_i = Polygon(contour_i)

        for j in range(i+1,n_objects):
            if j in suppressed_indexes:
                continue
            contour_j = sorted_contours[j]
            poly_j = Polygon(contour_j)

            polygon_intersection = poly_i.intersection(poly_j).area
            if polygon_intersection == 0:
                continue
            polygon_union = poly_i.union(poly_j).area
            IOU = polygon_intersection / polygon_union 
            
            if IOU > threshold:
                if not quiet:
                    print(f"({i},{j}): {IOU}")
                suppressed_indexes.append(j)

                if (output_name == ""):
                    continue
                if obs_color == 1:
                    cv2.fillPoly(nms_images,pts=[contour_i], color=1)
                obs_color += 10
                cv2.fillPoly(nms_images,pts=[contour_j], color=obs_color)
                
    
    if (output_name != ""):
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        gist_ncar = cm.get_cmap('gist_ncar', 256)
        newcolors = gist_ncar(np.linspace(0, 1, 256))
        black = np.array([0, 0, 0, 1])
        magneta = np.array([1, 0, 1, 1])
        newcolors[0, :] = black
        newcolors[255, :] = magneta
        newcmp = ListedColormap(newcolors)
        nuclei_cmap = newcmp
        plt.imshow(nms_images, cmap=nuclei_cmap, alpha=1)
        plt.savefig(f"{output_name.split('.')[0]}.png", bbox_inches = 'tight',pad_inches = 0)
        plt.close()
                
    for i in range(n_objects):
        if i not in suppressed_indexes:
            suppressed_contours.append(sorted_contours[i])
            suppressed_scores.append(sorted_scores[i])

    if not quiet:
        print(f"Suppressed objs by NMS: {n_objects-len(suppressed_scores)}")
    return suppressed_contours, suppressed_scores


def pred_to_tiff(objects,height,width,size_filter=0,score_filter=0.,nms_threshold=0.3,output_name="",nms_name=""):
    """
    Converts predicted objects to a TIFF image.

    Parameters:
        objects (list): A list of dictionaries representing the predicted objects.
        height (int): The height of the output image.
        width (int): The width of the output image.
        size_filter (int, optional): The minimum area required for an object to be included in the output. Defaults to 0.
        score_filter (float, optional): The minimum score required for an object to be included in the output. Defaults to 0.0.
        nms_threshold (float, optional): The threshold value for non-maximum suppression. Defaults to 0.3.
        output_name (str, optional): The name of the output TIFF file. Defaults to "".
        nms_name (str, optional): The name of the output file containing the non-maximum suppressed contours. Defaults to "".

    Returns:
        numpy.ndarray: A numpy array representing the resulting TIFF image.
    """
    # Create the numpy array
    masks = np.zeros((height, width), dtype=np.uint16)

    # For every object
    object_id = 1
    count = -1
    broken_poly_sum = 0

    contours = []
    areas = []
    scores = []
    for obj_dict in objects:
        count += 1
        height = obj_dict['segmentation']['size'][0]
        width = obj_dict['segmentation']['size'][1]
        score = obj_dict['score']

        # Score filter
        if score < score_filter:
            continue
        
        maskedArr = mask.decode(obj_dict['segmentation'])
        poly_list = polygon_from_mask(maskedArr)[0]
        if poly_list is None:
            broken_poly_sum += 1
            continue
        
        contour_list = [[poly_list[i], poly_list[i+1]] for i in range(0,len(poly_list),2) ]
        nd_contour = np.array(contour_list).astype("int64")
        poly = Polygon(nd_contour)

        # Area filter
        if poly.area < size_filter:
            continue

        contours.append(nd_contour)
        areas.append(poly.area)
        scores.append(score)
        object_id += 1
    
    # Filter objects with Non-Maximum Suppression algorithm
    s_contours, s_scores = non_maximum_suppression(contours,scores,height,width,nms_threshold,nms_name,1)
    s_contours = sort_objects_by(s_contours,s_scores)[0]
    
    object_id = 1
    for contour in s_contours:
        cv2.fillPoly(masks,pts=[contour], color=object_id)
        object_id += 1
    
    if broken_poly_sum > 0:
        print(f"Broken polygons on inference (less than 3 coordinate points): {broken_poly_sum}")

    if output_name != "":
        tifffile.imwrite(output_name, masks)
    return masks


def crop_fov(im,x1,x2,y1,y2):
    """
    Crop a region of interest from an image.

    Parameters:
        im (array-like): The input image.
        x1 (float): The starting x-coordinate of the region of interest.
        x2 (float): The ending x-coordinate of the region of interest.
        y1 (float): The starting y-coordinate of the region of interest.
        y2 (float): The ending y-coordinate of the region of interest.

    Returns:
        array-like: The cropped region of interest from the image.
    """
    return im[int(y1):int(y2), int(x1):int(x2)]


def filter_size(pred,min_area):
    """
    Filters the input array `pred` based on the size of connected components.

    Parameters:
        pred (ndarray): The input array.
        min_area (int): The minimum area threshold for connected components.

    Returns:
        ndarray: The filtered array.
    """
    ids = np.unique(pred)
    for i in range(len(ids)):
        pi = (pred == ids[i])
        area = np.sum(pi)
        if area < min_area:
            pred = pred*(pred != ids[i])
    return pred


def metrics(tp, fp, fn):
    """
    Calculates precision, recall, and F1-score based on true positives, false positives, and false negatives.

    Parameters:
        tp (int): The number of true positives.
        fp (int): The number of false positives.
        fn (int): The number of false negatives.

    Returns:
        p (float): The precision.
        r (float): The recall.
        f1 (float): The F1-score.

    If tp is 0, returns 0 for all values.
    """
    if tp == 0:
        return 0, 0, 0
    else:
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        f1 = 2*(p*r)/(p+r)
    return p,r,f1


def scatter_error_plot(x,y,xerr,yerr,color,xlabel,ylabel,name):
    """
    Generate a scatter error plot.

    Args:
        x (array-like): The x-coordinates of the data points.
        y (array-like): The y-coordinates of the data points.
        xerr (array-like): The x error values.
        yerr (array-like): The y error values.
        color (str): The color of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        name (str): The name of the output file.

    Returns:
        None
    """
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, color=color, ls='', marker='o')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(name)
    plt.close()


def histogram_plot(data,color,xlabel,ylabel,name):
    """
    Plots a histogram of the given data using the specified color.

    Parameters:
        data: A list of numerical values.
        color: A string representing the color of the histogram bars.
        xlabel: A string representing the label for the x-axis.
        ylabel: A string representing the label for the y-axis.
        name: A string representing the name of the output file.

    Returns:
        None
    """
    plt.hist(data, color=color)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(name)
    plt.close()


def scatter_plot(x,y,color,xlabel,ylabel,name,s=None):
    """
    Generate a scatter plot with the given data points.

    Parameters:
        x (list): The x-coordinates of the data points.
        y (list): The y-coordinates of the data points.
        color (str): The color of the data points.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        name (str): The name of the file to save the plot as.
        s (int, optional): The size of the data points. Defaults to None.
    """
    plt.scatter(x,y,color=color,s=s)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(name)
    plt.close()


def scatter_plot_with_regression(x,y,color,xlabel,ylabel,name):
    """
    Create a scatter plot with a regression line.

    Parameters:
        x (list): The x-coordinates of the data points.
        y (list): The y-coordinates of the data points.
        color (str): The color of the scatter plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        name (str): The name of the output file.

    Returns:
        None
    """
    df = pd.DataFrame(columns=[xlabel,ylabel])
    df[xlabel] = x
    df[ylabel] = y
    sns.lmplot(data=df, x=xlabel, y=ylabel, palette=[color])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(name)
    plt.close()


def inference_analysis_plots(results,output_path):
    """
    Generate inference analysis plots based on the results and save them to the specified output path.

    Parameters:
        results (list): A list of dictionaries containing the results of the inference analysis.
        output_path (str): The path where the generated plots will be saved.

    Returns:
        None
    """
    precision,recall,f1,objs,gb_ratio = [],[],[],[],[]
    for i in range(len(results)-1):
        precision.append(results[i]["precision"])
        recall.append(results[i]["recall"])
        f1.append(results[i]["f1"])
        objs.append(results[i]["gt_objects"])
        gb_ratio.append(results[i]["green_blue_ratio"])
    
    # Precision, recall and F1-score plots
    histogram_plot(precision,'orange','Precision','Counts',f"{output_path}/Precision_histogram.png")
    histogram_plot(recall,'orange','Recall','Counts',f"{output_path}/Recall_histogram.png")
    histogram_plot(f1,'orange','F1','Counts',f"{output_path}/F1-score_histogram.png")

    # Precision, recall and F1-score per number of objects and g/b ratio
    scatter_plot_with_regression(objs,precision,'orange','Number of objects','Precision',f"{output_path}/Precision_vs_n_objects.png")
    scatter_plot_with_regression(objs,recall,'orange','Number of objects','Recall',f"{output_path}/Recall_vs_n_objects.png")
    scatter_plot_with_regression(objs,f1,'orange','Number of objects','F1',f"{output_path}/F1-score_vs_n_objects.png")
    scatter_plot_with_regression(gb_ratio,precision,'orange','G/B Ratio','Precision',f"{output_path}/Precision_vs_gb-ratio.png")
    scatter_plot_with_regression(gb_ratio,recall,'orange','G/B Ratio','Recall',f"{output_path}/Recall_vs_gb-ratio.png")
    scatter_plot_with_regression(gb_ratio,f1,'orange','G/B Ratio','F1',f"{output_path}/F1-score_vs_gb-ratio.png")

    n_cells_bins = [0,100,200,300,400,500,600,700,800,900,1000]
    n_cells_x = [(n_cells_bins[i+1]-n_cells_bins[i])/2+n_cells_bins[i] for i in range(len(n_cells_bins)-1)]
    n_cells_xerr = [(n_cells_bins[i+1]-n_cells_bins[i])/2 for i in range(len(n_cells_bins)-1)]
    prec_n_cells = [None for c in range(len(n_cells_bins)-1)]
    recall_n_cells = [None for c in range(len(n_cells_bins)-1)]
    f1_n_cells = [None for c in range(len(n_cells_bins)-1)]
    for i in range(len(n_cells_bins)-1):
        tp,fp,fn=0,0,0
        for j in range(len(results)-1):
            if (results[j]["gt_objects"] >= n_cells_bins[i] and results[j]["gt_objects"] < n_cells_bins[i+1]):
                tp += results[j]["tp"]
                fp += results[j]["fp"]
                fn += results[j]["fn"]
        if tp > 0:
            prec_n_cells[i],recall_n_cells[i],f1_n_cells[i] = metrics(tp, fp, fn)
    
    scatter_error_plot(n_cells_x, prec_n_cells, n_cells_xerr, None, 'orange', 'Number of cells', 'Precision', f"{output_path}/Precision_vs_number-of-cells.png")
    scatter_error_plot(n_cells_x, recall_n_cells, n_cells_xerr, None, 'orange', 'Number of cells', 'Recall', f"{output_path}/Recall_vs_number-of-cells.png")
    scatter_error_plot(n_cells_x, f1_n_cells, n_cells_xerr, None, 'orange', 'Number of cells', 'F1', f"{output_path}/F1-score_vs_number-of-cells.png")


def create_inference_analysis(data_path,gt_annotations,inf_annotations,output_path,size_filter=25,score_filter=0.35,nms_threshold=0.3,quiet=True):
    """
    Creates an inference analysis for a given data set.

    Args:
        data_path (str): The path to the data set.
        gt_annotations (str): The path to the ground truth annotations file.
        inf_annotations (str): The path to the predicted annotations file.
        output_path (str): The path to the output directory.
        size_filter (int, optional): The size filter for the predicted annotations. Defaults to 25.
        score_filter (float, optional): The score filter for the predicted annotations. Defaults to 0.35.
        nms_threshold (float, optional): The NMS threshold for the predicted annotations. Defaults to 0.3.
        quiet (bool, optional): Whether to suppress console output. Defaults to True.

    Returns:
        list: A list of dictionaries containing the analysis results.
    """
    # GT infos
    print(f" - Get GT annotations")
    split_ids = get_split_ids(gt_annotations)
    pure_ids = [s[0] for s in split_ids]
    gt_anns = get_gt_annotations(gt_annotations,pure_ids)

    # Get predicted annotations
    print(f" - Get predicted annotations")
    pred_anns = get_prediction_annotations(inf_annotations)

    # Create folder for intersections
    intersections_path = f"{output_path}/intersections"
    os.system(f"mkdir {intersections_path}")

    # Create folder for intersections
    nms_path = f"{output_path}/NMS"
    os.system(f"mkdir {nms_path}")

    # Initialize variables
    sum_tp, sum_fp, sum_fn = 0, 0, 0
    results = []
    print(f" - Comparison Loop:")
    for (i,name) in split_ids:
        print(f"   > {i}: {name}")
        # Get the annotations for this image id
        gt_ann = gt_anns[str(i)]
        pred_ann = pred_anns[str(i)]

        # Transform them to 2D arrays
        gt = annotation_poly_to_tiff(gt_ann,gt_ann['images'][0]['height'],gt_ann['images'][0]['width'],ann_type="All")
        pred = pred_to_tiff(pred_ann,gt_ann['images'][0]['height'],gt_ann['images'][0]['width'],size_filter,score_filter,nms_threshold,nms_name=f"{nms_path}/{name}")
        im = skimage.io.imread(f"{data_path}/{name}")
        
        # Extract metrics
        iou = get_iou(pred,gt,quiet)
        tp, fp, fn = precision_at(0.7, iou,quiet)
        precision, recall, f1 = metrics(sum(tp), sum(fp), sum(fn))

        # Create intersection plots
        gt_per, gt_xor = get_intersections(gt,pred)
        file_path = f"{intersections_path}/intersections_{i}-{name.split('.')[0]}.png"
        plot_intersections(gt,pred,gt_xor,im,file_path)
        plt.close()
        if (not quiet):
            print(f"TP, FP, FN:  {sum(tp)}, {sum(fp)}, {sum(fn)}")
            print(f"pr, rc, f1:  {round(precision,2)}, {round(recall,2)}, {round(f1,2)}")

        sum_tp += sum(tp)
        sum_fp += sum(fp)
        sum_fn += sum(fn)

        r,g,b = get_image_channels(f"{data_path}/{name}")
        gb_ratio = get_channel_ratios(g,b)
        # Store results
        res = {
            "image":name,
            "image_id":i,
            "intersection_path":file_path,
            "gt_objects":len(np.unique(gt))-1,
            "detected_objects":len(np.unique(pred))-1,
            "green_blue_ratio":gb_ratio,
            "tp":int(sum(tp)),
            "fp":int(sum(fp)),
            "fn":int(sum(fn)),
            "precision":precision,
            "recall":recall,
            "f1":f1,
        }
        results.append(res)

    # Final metrics
    print(f"\n - Global metrics")
    gPrec, gRec, gF1 = metrics(sum_tp, sum_fp, sum_fn)
    print(f"TP, FP, FN: {sum_tp}, {sum_fp}, {sum_fn}")
    print(f"Precision:  {round(gPrec,2)}")
    print(f"Recall:     {round(gRec,2)}")
    print(f"F1 score:   {round(gF1,2)}")
    
    res = {
            "image":"all",
            "image_id":"",
            "intersection_path":"",
            "gt_objects":"",
            "detected_objects":"",
            "green_blue_ratio":None,
            "tp":int(sum_tp),
            "fp":int(sum_fp),
            "fn":int(sum_fn),
            "precision":gPrec,
            "recall":gRec,
            "f1":gF1,
        }
    results.append(res)
    with open(f"{output_path}/results.json", 'w') as fRes:
        json.dump(results, fRes)

    # Precision, recall, F1 and other plots
    inference_analysis_plots(results,output_path)

    return results


def inference_cluster_analysis(results,cluster_file,cluster_column,image_column,output_path):
    """
    Perform cluster analysis on the results of an inference.

    Parameters:
        results (list): A list of dictionaries containing the results of the inference.
        cluster_file (str): The path to the clustering file.
        cluster_column (str): The column name in the clustering file containing the cluster labels.
        image_column (str): The column name in the clustering file containing the image names.
        output_path (str): The path to the directory where the output files will be saved.

    Returns:
        DataFrame: A Pandas DataFrame containing the cluster metrics.
    """
    print(f" - Inference cluster analysis")
    # Extract information from results
    current_imgs = []
    results_per_image = {}
    for i in range(len(results)-1):
        current_imgs.append(results[i]['image'])
        results_per_image[results[i]['image']] = results[i]

    # Read clustering file and extract clusters
    df = pd.read_csv(cluster_file)
    clusters = df[cluster_column].unique()
    clusters.sort()

    # Loop over clusters
    p_arr,r_arr,f1_arr,c_arr,c_size,n_gts,n_gts_err,gb_ratio_arr,gb_ratio_arr_err = [],[],[],[],[],[],[],[],[]
    for c in clusters:
        # Extract cluster data as an array, filtered by the images we have available 
        cdf = df[df[cluster_column] == c]
        arr = np.array(cdf[cdf[image_column].isin(current_imgs)][image_column].to_list())
        size = len(arr)
        print(f"Size of cluster '{c}': {size}")
        if size == 0:
            continue

        cluster_path = f"{output_path}/cluster_{c}"
        os.system(f"mkdir {cluster_path}")
        
        # Extract TP, FP and FN per cluster
        tp,fp,fn,gts,gb_ratio = 0,0,0,[],[]
        for img in arr:
            tp += results_per_image[img]['tp']
            fp += results_per_image[img]['fp']
            fn += results_per_image[img]['fn']
            gts.append(results_per_image[img]['gt_objects'])
            gb_ratio.append(results_per_image[img]['green_blue_ratio'])
            os.system(f"cp {results_per_image[img]['intersection_path']} {cluster_path}")

        # Extract precision, recall and f1 per cluster
        precision,recall,f1 = metrics(tp,fp,fn)
        p_arr.append(precision)
        r_arr.append(recall)
        f1_arr.append(f1)
        c_arr.append(c)
        c_size.append(len(arr))
        n_gts.append(np.array(gts).mean())
        n_gts_err.append(np.array(gts).std())
        gb_ratio_arr.append(np.array(gb_ratio).mean())
        gb_ratio_arr_err.append(np.array(gb_ratio).std())
        print(f"Metrics (precis, recall, f1): {round(precision,2)}, {round(recall,2)}, {round(f1,2)}")
        print(f"GT Objects: {round(np.array(gts).mean(),2)}")
        print(f"G/B ratio: {round(np.array(gb_ratio).mean(),2)}")
    
    # Precision, recall and F1-score per cluster
    scatter_plot(c_arr, p_arr, 'orange', 'Cluster', 'Precision', f"{output_path}/Precision_vs_cluster.png",n_gts)
    scatter_plot(c_arr, r_arr, 'orange', 'Cluster', 'Recall', f"{output_path}/Recall_vs_cluster.png",n_gts)
    scatter_plot(c_arr, f1_arr, 'orange', 'Cluster', 'F1', f"{output_path}/F1-score_vs_cluster.png",n_gts)
    scatter_error_plot(c_arr, n_gts, None, n_gts_err, 'orange', 'Cluster', 'GT Objects', f"{output_path}/GT-objects_vs_cluster.png")
    scatter_error_plot(c_arr, gb_ratio_arr, None, gb_ratio_arr_err, 'orange', 'Cluster', 'G/B Ratio', f"{output_path}/GB-ratio_vs_cluster.png")

    df = pd.DataFrame(columns=['cluster','precision','recall','f1'])
    df['cluster'] = c_arr
    df['precision'] = p_arr
    df['recall'] = r_arr
    df['f1'] = f1_arr

    return df


def get_channel_ratios(channel_1,channel_2,from_counts=False):
    """
    Calculate the channel ratios between two given channels.

    Parameters:
        channel_1 (array-like): The first channel.
        channel_2 (array-like): The second channel.
        from_counts (bool, optional): If True, calculate the ratio from the counts of the channels. Defaults to False.

    Returns:
        float: The channel ratio.
    """
    if from_counts:
        return np.sum(np.array(channel_1).astype(bool))/np.sum(np.array(channel_2).astype(bool))
    return np.sum(np.array(channel_1))/np.sum(np.array(channel_2))


def get_image_channels(file):
    """
    Open an image file and return its individual RGB channels.

    Parameters:
        file (str): The path to the image file.

    Returns:
        tuple: A tuple containing three image channels: red, green, and blue.
    """
    img = Image.open(file, mode='r')
    img = img.convert('RGB')
    r,g,b = img.split()
    return r,g,b