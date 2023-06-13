import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import urllib.request as urllib
import pycocotools.mask as mask
import json
import cv2
import tifffile
import skimage
from shapely.geometry import Polygon
from skimage.color import rgb2gray
from split import get_split_ids


def get_iou(inference, gt, quiet=True):
    """
    Returns a 2D float for intersection over union ratio between ground truth and inference labels.
    
    Parameters
    ----------
    inference : 2D numpy array (uint16)
        The inference labels.

    gt : 2D numpy array (uint16)
        The ground truth labels.

    quiet : boolean
        Option to output information while running.
    
    Returns
    -------
    iou : 2D numpy array (float)
        The intersection over union ratio between all pairs of inference and ground truth labels.
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
    Returns the count of true positive, false positives, false negatives at a given threshold
    for iou.
    
    Parameters
    ----------
    threshold: float
        The intersection over union threshold to consider a true positive prediction

    iou : 2D numpy (float)
        The intersection over union for all pair of ground truth and inference lables

    quiet : boolean
        Option to output information while running.
    
    Returns
    -------
    true_positives : numpy array (boolean)
        Array of true positives boolean values between the gt and inference comparison.

    false_positives : numpy array (boolean)
        Array of false positives boolean values between the gt and inference comparison.

    false_negatives : numpy array (boolean)
        Array of false negatives boolean values between the gt and inference comparison.
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
    Returns a lookup table after permuting the IDs.
    
    Parameters
    ----------
    ids: 2D numpy array
        The 2D IDs array.
    
    Returns
    -------
    lookup : 1D numpy array
        The permuted IDs.
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
    Returns the intersections between ground truth and inference labels.
    
    Parameters
    ----------
    gt : 2D numpy array (uint16)
        The ground truth labels.

    pred : 2D numpy array (uint16)
        The inference labels.
    
    Returns
    -------
    inference_permuted : 2D numpy array
        The permuted inference.

    gt_xor_inference : 2D numpy array
        The False Positive objects (XOR) + False Negative objexts (FN perm) between inference
        and ground truth labels.
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


def plot_intersections(gt,gt_per,gt_xor,image,name='intersections.png'):
    """
    Create the plots of the intersections between ground truth and inference labels.
    
    Parameters
    ----------
    gt : 2D numpy array
        The ground truth masks.

    gt_per : 2D numpy array
        The inference masks.

    gt_xor : 2D numpy array
        The false positive/negative masks.
    
    image : 2D numpy array
        The experimental image.

    name : string
        Name of the output file.
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


def get_prediction_annotations(file):
    """
    Load the inference annotations into a dictionary where the key is the image id.
    
    Parameters
    ----------
    file : string
        Inference annotation file in coco format.
    
    Returns
    -------
    pred : dictionary
        Inference annotation for all of the images, having the key as the image id.
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
    Load the ground truth annotations into a dictionary where the key is the image id, and the ids
    are the image ids that should be retrieved from the annotation file.
    
    Parameters
    ----------
    file : string
        Inference annotation file in coco format.

    ids : string
        The image ids within the annotation file that must be retrieved.
    
    Returns
    -------
    gt : dictionary
        Ground truth annotation for all of the images within the ids, having the key as the image id.
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


def annotation_poly_to_tiff(json_data,height,width,name="",ann_type="All"):
    """
    Converts the annotation of a dictionary into a tiff format (arrays).
    
    Parameters
    ----------
    json_data : dictionary
        Dictionary containing all of the annotations.

    height : int
        The image height.

    width : int
        The image width.

    name : string
        The output file name to save the masks.

    ann_type : string
        The image ids within the annotation file that must be retrieved.
    
    Returns
    -------
    masks : 2D array
        Annotations in a single matrix.
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
    
    if name != "":
        tifffile.imwrite(name, masks)
    return masks


def polygonFromMask(maskedArr): # https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    """
    Converts a mask annotation into polygon format.
    
    Parameters
    ----------
    maskedArr : 2D array
        Matrix containing the mask.
    
    Returns
    -------
    polygon : 1D array
        Polygon list of coordinates.

    bbox : array
        Boundary box for the annotation.

    area : int
        Area of the mask.
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
    # RLE = mask.encode(np.asfortranarray(maskedArr))
    area = mask.area(RLE)
    [x, y, w, h] = cv2.boundingRect(maskedArr)

    return segmentation[0], [x, y, w, h], area

def pred_to_tiff(objects,height,width,name=""):
    """
    Converts the annotation of an object dictionary into a tiff format (arrays).
    
    Parameters
    ----------
    objects : dictionary
        Dictionary containing all of the annotations.

    height : int
        The image height.

    width : int
        The image width.

    name : string
        The output file name to save the masks.
    
    Returns
    -------
    masks : 2D array
        Annotations in a single matrix.
    """
    # Create the numpy array
    masks = np.zeros((height, width), dtype=np.uint16)

    # For every object
    object_id = 1
    count = -1
    sum_ = 0

    contours = []
    areas = []
    for obj_dict in objects:
        count += 1
        height = obj_dict['segmentation']['size'][0]
        width = obj_dict['segmentation']['size'][1]
        
        maskedArr = mask.decode(obj_dict['segmentation'])
        poly_list = polygonFromMask(maskedArr)[0]
        if poly_list is None:
            sum_ += 1
            continue
        
        contour_list = [[poly_list[i], poly_list[i+1]] for i in range(0,len(poly_list),2) ]
        nd_contour = np.array(contour_list).astype("int64")
        contours.append(nd_contour)
        poly = Polygon(nd_contour)
        areas.append(poly.area)
        object_id += 1
    
    # TODO: Replace code below by Non-Maximum Suppression algorithm
    sorted_contours = contours.copy()
    sorted_areas = areas.copy()
    for i in range(len(sorted_areas)):
        min = np.min(sorted_areas[i:])
        stop = False
        for j in range(i,len(sorted_areas)):
            if stop:
                continue
            if sorted_areas[j] == min:
                val = sorted_areas[i]
                sorted_areas[i] = sorted_areas[j]
                sorted_areas[j] = val
                cont = sorted_contours[i].copy()
                sorted_contours[i] = sorted_contours[j].copy()
                sorted_contours[j] = cont.copy()
                stop = True

    object_id = 1
    for contour in sorted_contours:
        cv2.fillPoly(masks,pts=[contour], color=object_id)
        object_id += 1
    
    print(f"Broken polygons on inference (less than 3 coordinate points): {sum_}")
    if name != "":
        tifffile.imwrite(name, masks)
    return masks


def crop_fov(im,x1,x2,y1,y2):
    """
    Returns a cropped FOV based on input parameters.
    
    Parameters
    ----------
    im : 2D numpy array
        Image information.

    x1 : int
        Initial pixel crop on x-axis.

    x2 : int
        Final pixel crop on x-axis.

    y1 : int
        Initial pixel crop on y-axis.

    y2 : int
        Final pixel crop on y-axis.
    
    Returns
    -------
    cropped image : 2D numpy array
        The cropped FOV.
    """
    return im[int(y1):int(y2), int(x1):int(x2)]


def filter_size(pred,min_area):
    """
    Filter the annotations based on an area.
    
    Parameters
    ----------
    pred : 2D numpy array
        Inference annotations.

    min_area : int
        Minimum acceptable area (px^2).
    
    Returns
    -------
    pred : 2D numpy array
        Filtered inference annotations.
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
    Returns the precision, recall and F1 score based on the number of
    true positives, false positives and false negatives.
    
    Parameters
    ----------
    tp : int
        Number of true positives.

    fp : int
        Number of false positives.

    fn : int
        Number of false negatives.
    
    Returns
    -------
    p : float
        Precision.

    r : float
        Recall.

    f1 : float
        F1-score.
    """
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    f1 = 2*(p*r)/(p+r)
    return p,r,f1


def create_inference_analysis(data_path,gt_annotations,inf_annotations,output_path,size_filter=25,quiet=True):
    """
    Analyze the performance of the object detection agains the ground truth objects of a completed dataset. Create
    the plots of the intersections between the GT and the inference objects and return the results per image.
    
    Parameters
    ----------
    data_path : string
        Folder path where the images are located.

    gt_annotations : string
        File path containing the ground truth annotations for all of the images in coco format.

    inf_annotations : string
        File path containing the inference annotations for all of the images in coco format.

    output_path : string
        Folder path to store the output files generate during the analysis.

    size_filter : int
        Filter objecs smaller than this area (px^2).

    quiet : boolean
        Option to output information while running.
    
    Returns
    -------
    results : array(dictionary)
        An array of dictionaries where each entry has stored the image name, the image id, and the metrics.
    """
    # GT infos
    print(f" - Get GT annotations")
    split_ids = get_split_ids(gt_annotations)
    pure_ids = [s[0] for s in split_ids]
    gt_anns = get_gt_annotations(gt_annotations,pure_ids)

    # Get predicted annotations
    print(f" - Get predicted annotations")
    pred_anns = get_prediction_annotations(inf_annotations)

    # Initialize variables
    sum_tp, sum_fp, sum_fn = 0, 0, 0
    all_prec, all_rec, all_f1 = [], [], []
    results = []
    print(f" - Comparison Loop:")
    for (i,name) in split_ids[:2]:
        print(f"   > {i}: {name}")
        # Get the annotations for this image id
        gt_ann = gt_anns[str(i)]
        pred_ann = pred_anns[str(i)]

        # Transform them to 2D arrays
        gt = annotation_poly_to_tiff(gt_ann,gt_ann['images'][0]['height'],gt_ann['images'][0]['width'],name="",ann_type="All")
        pred = pred_to_tiff(pred_ann,gt_ann['images'][0]['height'],gt_ann['images'][0]['width'],name="")
        im = skimage.io.imread(f"{data_path}/{name}")
        
        # Filter small ones
        pred = filter_size(pred,size_filter)
        
        # Extract metrics
        iou = get_iou(pred,gt,quiet)
        tp, fp, fn = precision_at(0.7, iou,quiet)
        precision, recall, f1 = metrics(sum(tp), sum(fp), sum(fn))
        all_prec.append(precision)
        all_rec.append(recall)
        all_f1.append(f1)

        # Create intersection plots
        gt_per, gt_xor = get_intersections(gt,pred)
        plot_intersections(gt,pred,gt_xor,im,f"{output_path}/intersections_{i}.png")
        plt.close()
        if (not quiet):
            print(f"TP, FP, FN:  {sum(tp)}, {sum(fp)}, {sum(fn)}")
            print(f"pr, rc, f1:  {round(precision,2)}, {round(recall,2)}, {round(f1,2)}")

        sum_tp += sum(tp)
        sum_fp += sum(fp)
        sum_fn += sum(fn)

        # Store results
        res = {
            "image":name,
            "image_id":i,
            "tp":sum(tp),
            "fp":sum(fp),
            "fn":sum(fn),
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
            "image_id":None,
            "tp":sum_tp,
            "fp":sum_fp,
            "fn":sum_fn,
            "precision":gPrec,
            "recall":gRec,
            "f1":gF1,
        }
    results.append(res)

    # Precision, recall and F1 plots
    plt.hist(all_prec, color='orange')
    plt.ylabel('Counts')
    plt.xlabel('Precision')
    plt.savefig(f"{output_path}/global_precision.png")
    plt.close()

    plt.hist(all_rec, color='orange')
    plt.ylabel('Counts')
    plt.xlabel('Recall')
    plt.savefig(f"{output_path}/global_recall.png")
    plt.close()

    plt.hist(all_f1, color='orange')
    plt.ylabel('Counts')
    plt.xlabel('F1')
    plt.savefig(f"{output_path}/global_f1-score.png")
    plt.close()

    return results