import os, math, json, random
import numpy as np
import pandas as pd
import cv2
from albumentations import RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip, Defocus, Blur
from albumentations.augmentations.crops.transforms import RandomCrop
from pyoseg.inference import annotation_poly_to_tiff, plot_gt_image
from PIL import Image
import pycocotools.mask as cocomask

augmentations = {
    "HorizontalFlip": HorizontalFlip(p=1.),
    "VerticalFlip": VerticalFlip(p=1.),
    "RandomRotate90": RandomRotate90(p=1.),
    "GridDistortion": GridDistortion(p=1.),
    "Defocus": Defocus(p=1.),
    "Blur": Blur(p=1.),
    #"RandomCrop":RandomCrop(p=1.,height=256,width=256)
}


def get_images_and_masks(data_path, annotations_path, annotation_suffix = "_coco.json"):
    """
    Get a list of images and masks from the given data path and annotations path.

    Parameters:
        data_path (str): The path to the directory containing the images.
        annotations_path (str): The path to the directory containing the masks.
        annotation_suffix (str, optional): The suffix used to identify the annotation files.

    Returns:
        tuple: A tuple containing three lists - images, masks, and extension.
               - images (list): A list of image filenames without the extension.
               - masks (list): A list of mask filenames without the annotation suffix.
               - extension (str): The extension of the image files.

    Raises:
        AssertionError: If no images are found in the 'data_path' directory or no masks are found in the 'annotations_path' directory.
        AssertionError: If images cannot be correlated with annotations or annotations cannot be correlated with images.
        AssertionError: If the number of images is not the same as the number of masks.
"""
    images = [f for f in os.listdir(data_path)]
    masks = [f[:-len(annotation_suffix)] for f in os.listdir(annotations_path) if annotation_suffix in f]
    assert len(images) > 0, f"No images found inside '{data_path}', please check."
    assert len(masks) > 0,  f"No images found inside '{annotations_path}', please check."

    print(f"Number of images on '{data_path}':\n{len(images)}")
    print(f"Number of masks on '{annotations_path}':\n{len(masks)}")

    extension = "." + images[0].split('.')[-1]
    images_to_rm = []
    for i in range(len(images)):
        if images[i][:-len(extension)] not in masks:
            images_to_rm.append(i)
    images = [images[i][:-len(extension)] for i in range(len(images)) if i not in images_to_rm]
    if len(images_to_rm) > 0:
        print(f"It was found {len(images_to_rm)} images without related annotations.")

    masks_to_rm = []
    for i in range(len(masks)):
        if masks[i] not in images:
            masks_to_rm.append(i)
    masks = [masks[i] for i in range(len(masks)) if i not in masks_to_rm]
    if len(masks_to_rm) > 0:
        print(f"It was found {len(masks_to_rm)} masks without related image.")

    assert len(images) > 0, f"Images could not be correlated with annotations, please check."
    assert len(masks) > 0,  f"Annotations could not be correlated with images, please check."
    assert len(images) == len(masks), f"Problem in correlated images and masks."

    print(f"Current dataset size: {len(images)}")
    print(f"Data extension: {extension}")
    print(f"Mask suffix:    {annotation_suffix}")

    return images, masks, extension


def augment_data(data_path, annotations_path, output_path, annotation_suffix = "_coco.json", functions = augmentations.keys()):
    """
    Augments data by applying a list of specified augmentation functions to the images and masks.

    Parameters:
        data_path (str): Path to the directory containing the images.
        annotations_path (str): Path to the directory containing the masks.
        output_path (str): Path to the directory where the augmented images and masks will be saved.
        annotation_suffix (str, optional): Suffix to be appended to the mask filenames. Defaults to "_coco.json".
        functions (list, optional): List of augmentation functions to be applied. Defaults to ["RandomCrop"].

    Raises:
        AssertionError: If any of the specified augmentation functions are invalid.

    Returns:
        None
    """
    assert [i in functions for i in augmentations.keys()], "Please provide a valid augmentation function."

    images, masks, extension = get_images_and_masks(data_path, annotations_path, annotation_suffix)
    dataset_size = len(images)

    for i in range(dataset_size):
        print(f"Augmenting image {i+1}/{dataset_size}:")
        image_i = images[i] + extension
        mask_i = images[i] + annotation_suffix

        image = cv2.imread(f"{data_path}/{image_i}", cv2.IMREAD_COLOR)
        mask0 = read_annotation_file(f"{annotations_path}/{mask_i}")

        n_aug = len(functions)
        mask0["images"][0]['file_name'] = f"{images[i]}_aug{n_aug}{extension}"

        cv2.imwrite(f"{output_path}/{images[i]}_aug{n_aug}{extension}", image)
        save_mask(f"{output_path}/{images[i]}_aug{n_aug}{annotation_suffix}", mask0)
        gt_image = Image.fromarray(np.uint8(image.copy()))
        gt = annotation_poly_to_tiff(mask0,mask0['images'][0]['height'],mask0['images'][0]['width'],ann_type="All")
        plot_gt_image(gt_image,gt,f"{output_path}/GT_{images[i]}_aug{n_aug}.png")

        height, width, channels = image.shape
        output_images, output_masks = [],[]
        masks0 = mask_array(mask0)

        for f in functions:
            masks = masks0.copy()
            augmented = augmentations[f](image=image, masks=masks)
            output_images.append(augmented['image'])
            output_masks.append(augmented['masks'])

        idx = 0
        for i_image, i_masks in zip(output_images, output_masks):
            tmp_img_name = f"{output_path}/{images[i]}_aug{idx}{extension}"
            tmp_mask_name = f"{output_path}/{images[i]}_aug{idx}{annotation_suffix}"
            tmp_gt_name = f"{output_path}/GT_{images[i]}_aug{idx}.png"

            cv2.imwrite(tmp_img_name, i_image)
            mask = mask0.copy()
            mask["images"][0]['file_name'] = f"{images[i]}_aug{idx}{extension}"
            mask["images"][0]['height'] = i_image.shape[0]
            mask["images"][0]['width'] = i_image.shape[1]

            mask["annotations"] = []
            for mask_id in range(len(i_masks)):
                segmentation = cv_to_poly(i_masks[mask_id])
                if segmentation is None:
                    continue
                area = get_area_from_poly(segmentation)
                bbox = get_bbox_from_poly(segmentation)
                new_mask = {
                    "id":mask_id,
                    "image_id":mask["images"][0]["id"],
                    "category_id":1,
                    "iscrowd": 0,
                    "segmentation":[segmentation],
                    "area":area,
                    "bbox":bbox}
                mask["annotations"].append(new_mask)
            save_mask(tmp_mask_name, mask)
            
            gt = annotation_poly_to_tiff(mask,mask['images'][0]['height'],mask['images'][0]['width'],ann_type="All")
            plot_gt_image(i_image,gt,tmp_gt_name)
            idx += 1


def save_mask(mask_path, mask): 
    """
    Save a mask to a file.

    Parameters:
        mask_path (str): The path to save the mask file.
        mask (object): The mask object to be saved.
    """
    with open(mask_path, 'w') as ann:
        json.dump(mask, ann)


def poly_to_cv(poly,height,width,color=1):
    """
    Generates a binary mask image in OpenCV format from a polygon.

    Parameters:
        poly (list): A list of polygon coordinates in the format [x1, y1, x2, y2, ...].
        height (int): The height of the output mask.
        width (int): The width of the output mask.

    Returns:
        cv2_mask (ndarray): A binary mask image with shape (height, width).
    """
    cv2_mask = np.zeros((height, width), dtype=np.uint8)
    contour_list = [[poly[i*2], poly[i*2+1]] for i in range(int(len(poly)/2))]
    nd_contour = np.array(contour_list).astype("int64")
    cv2.fillPoly(cv2_mask, [nd_contour], color)
    return cv2_mask


def cv_to_poly(cv_mask):
    """
    Converts a contour mask to a polygon representation.
    
    Parameters:
        cv_mask (numpy.ndarray): The contour mask to be converted.
    
    Returns:
        list or None: The polygon representation of the contour mask, or None if the contour mask does not meet the size requirement.
    """
    if np.sum(cv_mask) == 0:
        return None
    contours, _ = cv2.findContours(cv_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    contour = contours[0]
    if contour.size >= 6:
        return contour.flatten().tolist()
    return None


def read_annotation_file(file):
    """
    Read an annotation file and return the loaded data.

    Parameters:
        file (str): The path to the annotation file.

    Returns:
        dict: The loaded annotation data.
    """
    with open(file) as f:
        ann = json.load(f)
    return ann


def mask_array(mask):
    """
    Generate an array of masks from a given mask object.

    Parameters:
        mask (dict): The mask object containing annotations and image dimensions.

    Returns:
        list: An array of masks converted from the annotations in the mask object.
    """
    height,width = mask['images'][0]['height'], mask['images'][0]['width']
    masks = []
    for i in range(len(mask['annotations'])):
        poly_list = mask['annotations'][i]['segmentation'][0]
        cv_mask = poly_to_cv(poly_list,height,width)
        masks.append(cv_mask)
    return masks


def random_crop(data_path, annotations_path, output_path, annotation_suffix="_coco.json", size=(256, 256, 10)):
    """
    This function creates an extended dataset with random crops.
    
    Parameters:
        data_path (str): Path where the images are located.
        annotations_path (str): Path where the coco annotations are located.
        output_path (str): Path to the output folder.
        annotation_suffix (str): The suffix to the coco annotations in comparison to the image it is related. Default is "_coco.json".
        size (tuple): The crop size and number of crops: (x_width, y_height, n_crops).
    """

    # Get images, masks, and extension
    images, masks, extension = get_images_and_masks(data_path, annotations_path, annotation_suffix)

    # Get dataset size
    dataset_size = len(images)

    # Iterate over the dataset
    for i in range(dataset_size):
        image_i = images[i] + extension
        mask_i = images[i] + annotation_suffix

        # Read image and mask
        image = cv2.imread(f"{data_path}/{image_i}", cv2.IMREAD_COLOR)
        mask = read_annotation_file(f"{annotations_path}/{mask_i}")
        mask["images"][0]['file_name'] = f"{images[i]}_crop{size[2]}{extension}"

        cv2.imwrite(f"{output_path}/{images[i]}_crop{size[2]}{extension}", image)
        save_mask(f"{output_path}/{images[i]}_crop{size[2]}{annotation_suffix}", mask)
        gt_image = Image.fromarray(np.uint8(image.copy()))
        gt = annotation_poly_to_tiff(mask,mask['images'][0]['height'],mask['images'][0]['width'],ann_type="All")
        plot_gt_image(gt_image,gt,f"{output_path}/GT_{images[i]}_crop{size[2]}.png")

        # Generate random crops
        for j in range(size[2]):
            xi, yi = random.randint(0, image.shape[1] - size[1]), random.randint(0, image.shape[0] - size[0])
            xf, yf = xi + size[1], yi + size[0]
            
            cropped_image = crop_image(image, xi, xf, yi, yf)
            mask = read_annotation_file(f"{annotations_path}/{mask_i}")
            cropped_mask = crop_mask(mask, image.shape[0], image.shape[1], xi, xf, yi, yf)

            if len(cropped_mask['annotations']) == 0:
                j -= 1
                continue

            output_image_name = f"{output_path}/{images[i]}_crop{j}{extension}"
            output_mask_name = f"{output_path}/{images[i]}_crop{j}{annotation_suffix}"
            output_gt_name = f"{output_path}/GT_{images[i]}_crop{j}.png"

            cropped_mask["images"][0]['file_name'] = f"{images[i]}_crop{j}{extension}"
            cropped_mask['images'][0]['height'] = size[1]
            cropped_mask['images'][0]['width'] = size[0]
            
            cv2.imwrite(output_image_name, cropped_image)
            save_mask(output_mask_name, cropped_mask)

            gt_image = Image.fromarray(np.uint8(cropped_image.copy()))
            gt = annotation_poly_to_tiff(cropped_mask,cropped_mask['images'][0]['height'],cropped_mask['images'][0]['width'],ann_type="All")
            plot_gt_image(gt_image,gt,output_gt_name)



def crop_mask(mask,height,width,xi,xf,yi,yf):
    """
    Crop the given mask according to the specified dimensions and return the updated mask.
    
    Parameters:
        mask (dict): The mask to be cropped.
        height (int): The height of the mask.
        width (int): The width of the mask.
        xi (int): The starting x-coordinate of the cropping region.
        xf (int): The ending x-coordinate of the cropping region.
        yi (int): The starting y-coordinate of the cropping region.
        yf (int): The ending y-coordinate of the cropping region.
    
    Returns:
        dict: The cropped mask.
    """
    cropped_mask = mask.copy()
    to_remove = []
    for i in range(len(cropped_mask['annotations'])):
        # Get polygons
        poly_list = cropped_mask['annotations'][i]['segmentation'][0]
        cv_mask = poly_to_cv(poly_list,height,width)

        # Apply crop
        cv_mask = crop_image(cv_mask,xi,xf,yi,yf)
        
        # Return to coco format
        cropped_annotation = cv_to_poly(cv_mask)
        if cropped_annotation is not None:
            cropped_mask['annotations'][i]['segmentation'][0] = cropped_annotation
            cropped_mask['annotations'][i]['area'] = get_area_from_poly(cropped_annotation)
            cropped_mask['annotations'][i]['bbox'] = get_bbox_from_poly(cropped_annotation)
        else:
            to_remove.append(i)

    for i in reversed(to_remove):
        del cropped_mask['annotations'][i]

    return cropped_mask


def get_bbox_from_poly(poly):
    """
    Calculate the bounding box coordinates from a polygon.

    Parameters:
        poly (list): A list of coordinates representing a polygon. The coordinates are in the form of [x1, y1, x2, y2, ..., xn, yn].

    Returns:
        list: A list of four values representing the bounding box coordinates. The values are in the form of [min_x, min_y, width, height].
    """
    poly_x = [poly[i] for i in range(0,len(poly),2)]
    poly_y = [poly[i] for i in range(1,len(poly),2)]
    min_x = min(poly_x)
    min_y = min(poly_y)
    max_x = max(poly_x)
    max_y = max(poly_y)
    width = max_x - min_x
    height = max_y - min_y
    return [min_x, min_y, width, height]


def get_area_from_poly(poly):
    """
    Calculate the area of a polygon.

    Parameters:
        poly: a list of points representing the polygon.

    Returns:
        int: the area of the polygon as an integer.
    """
    return int(np.sum(np.array(poly,bool)))


def crop_image(image,xi,xf,yi,yf):
    """
    Crop a given image using the specified coordinates.

    Parameters:
        image (numpy.ndarray): The input image.
        xi (int): The starting x-coordinate of the crop.
        xf (int): The ending x-coordinate of the crop.
        yi (int): The starting y-coordinate of the crop.
        yf (int): The ending y-coordinate of the crop.

    Returns:
        numpy.ndarray: The cropped image.
    """
    return image[yi:yf, xi:xf]