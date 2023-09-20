import os
import json
import random
import numpy as np
import cv2
import albumentations as A
from albumentations import RandomRotate90, GridDistortion, HorizontalFlip, \
    VerticalFlip, Blur, RandomBrightnessContrast, RandomGamma, RandomCrop
from pyoseg.inference import annotation_poly_to_tiff, plot_gt_image, cv_to_poly, \
    poly_to_cv, get_poly_from_segmentation
from PIL import Image
import gc
import shapely


augmentations = {
    "RandomCrop": RandomCrop(p=1., height=256, width=256),
    "HorizontalFlip": HorizontalFlip(p=.5),
    "VerticalFlip": VerticalFlip(p=.5),
    "RandomRotate90": RandomRotate90(p=.5),
    "GridDistortion": GridDistortion(
        p=.7, normalized=True, distort_limit=0.3),
    "Blur": Blur(p=.1, blur_limit=(3, 7)),
    "RandomBrightnessContrast": RandomBrightnessContrast(
        p=.5, brightness_limit=0.2, contrast_limit=0.2),
    "RandomGamma": RandomGamma(p=.5, gamma_limit=(80, 120))
}


def get_images_and_masks(
        data_path, annotations_path, annotation_suffix="_coco.json"):
    """
    Get a list of images and masks from the given data path and annotations
    path.

    Parameters:
        data_path (str): The path to the directory containing the images.
        annotations_path (str): The path to the directory containing the
            masks.
        annotation_suffix (str, optional): The suffix used to identify the
            annotation files.

    Returns:
        tuple: A tuple containing three lists - images, masks, and extension.
            - images (list): A list of image filenames without the extension.
            - masks (list): A list of mask filenames without suffix.
            - extension (str): The extension of the image files.
"""
    images = [f for f in os.listdir(data_path) if not f.startswith(".")]
    masks = [f[:-len(annotation_suffix)] for f in
             os.listdir(annotations_path) if annotation_suffix in f and not f.startswith(".")]
    if len(images) == 0:
        raise FileNotFoundError(
            f"No images found inside '{data_path}', please check.")
    if len(masks) == 0:
        raise FileNotFoundError(
            f"No images found inside '{annotations_path}', please check.")

    print(f"Number of images on '{data_path}':\n{len(images)}")
    print(f"Number of masks on '{annotations_path}':\n{len(masks)}")

    extension = "." + images[0].split('.')[-1]
    images_to_rm = []
    for i in range(len(images)):
        if images[i][:-len(extension)] not in masks:
            images_to_rm.append(i)
    images = [images[i][:-len(extension)] for i in range(len(images))
              if i not in images_to_rm]
    if len(images_to_rm) > 0:
        print(f"{len(images_to_rm)} images without related annotations.")

    masks_to_rm = []
    for i in range(len(masks)):
        if masks[i] not in images:
            masks_to_rm.append(i)
    masks = [masks[i] for i in range(len(masks)) if i not in masks_to_rm]
    if len(masks_to_rm) > 0:
        print(f"{len(masks_to_rm)} masks without related image.")

    if len(images) == 0:
        raise ValueError(
            "Images could not be correlated with annotations, please check.")
    if len(masks) == 0:
        raise ValueError(
            "Annotations could not be correlated with images, please check.")
    if len(masks) != len(images):
        raise ValueError(
            f"Problem in correlated images and masks. Number differs: \
            {len(images)} vs {len(masks)}.")

    print(f"Current dataset size: {len(images)}")
    print(f"Data extension: {extension}")
    print(f"Mask suffix:    {annotation_suffix}")

    return images, masks, extension


def create_augmentation_transform(functions=augmentations.keys()):
    """
    Create an augmentation model using a list of selected augmentation
    functions.

    Parameters:
        functions (list): A list of augmentation function names. Defaults
            to all available augmentation functions.

    Returns:
        transform: The composed augmentation transform.
    """
    for f in functions:
        if f not in augmentations.keys():
            raise KeyError(
                f"Please provide a valid augmentation function: {f}.")
    return A.Compose([augmentations[f] for f in functions])


def augment_data(
        data_path, annotations_path, output_path,
        annotation_suffix="_coco.json", functions=augmentations.keys(),
        times=10):
    """
    Augments data by applying a list of specified augmentation functions to
    the images and masks.

    Parameters:
        data_path (str): Path to the directory containing the images.
        annotations_path (str): Path to the directory containing the masks.
        output_path (str): Path to the directory where the augmented images
            and masks will be saved.
        annotation_suffix (str, optional): Suffix to be appended to the mask
            filenames. Defaults to "_coco.json".
        functions (list, optional): List of augmentation functions to be
            applied. Defaults to ["RandomCrop"].
        times (int, optional): Number of times to apply the augmentation
            functions. Defaults to 10.

    Returns:
        None
    """
    assert [i in functions for i in augmentations.keys()], \
        "Please provide a valid augmentation function."
    transform = create_augmentation_transform(functions)

    images, masks, extension = get_images_and_masks(
        data_path, annotations_path, annotation_suffix)
    dataset_size = len(images)

    for i in range(dataset_size):
        print(f"Augmenting image {i+1}/{dataset_size}:")
        image_i = images[i] + extension
        mask_i = images[i] + annotation_suffix

        image = cv2.imread(f"{data_path}/{image_i}", cv2.IMREAD_COLOR)
        mask0 = read_annotation_file(f"{annotations_path}/{mask_i}")

        n_aug = times
        mask0["images"][0]['file_name'] = f"{images[i]}_aug{n_aug}{extension}"

        cv2.imwrite(f"{output_path}/{images[i]}_aug{n_aug}{extension}", image)
        save_mask(
            f"{output_path}/{images[i]}_aug{n_aug}{annotation_suffix}", mask0)
        gt_image = Image.fromarray(np.uint8(image.copy()))
        gt = annotation_poly_to_tiff(
            mask0, mask0['images'][0]['height'], mask0['images'][0]['width'],
            ann_type="All")
        #plot_gt_image(
        #    gt_image, gt, f"{output_path}/GT_{images[i]}_aug{n_aug}.png")

        has_masks = np.sum(gt) != 0
        if not has_masks:
            print(f"Image {image_i} has no masks.")

        height, width, channels = image.shape
        output_images, output_masks = [], []
        masks0 = mask_array(mask0)

        for t in range(times):
            augmented = None
            if has_masks:
                masks = masks0.copy()
                augmented = transform(image=image, masks=masks)
                output_masks.append(augmented['masks'])
            else:
                augmented = transform(image=image)
            output_images.append(augmented['image'])

        idx = 0
        for i_image, i_masks in zip(output_images, output_masks):
            tmp_img_name = \
                f"{output_path}/{images[i]}_aug{idx}{extension}"
            tmp_mask_name = \
                f"{output_path}/{images[i]}_aug{idx}{annotation_suffix}"
            tmp_gt_name = \
                f"{output_path}/GT_{images[i]}_aug{idx}.png"

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
                    "id": mask_id,
                    "image_id": mask["images"][0]["id"],
                    "category_id": 1,
                    "iscrowd": 0,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox}
                mask["annotations"].append(new_mask)
            save_mask(tmp_mask_name, mask)

            gt = annotation_poly_to_tiff(
                mask, mask['images'][0]['height'],
                mask['images'][0]['width'], ann_type="All")
            #plot_gt_image(i_image, gt, tmp_gt_name)
            idx += 1

        del image, mask0, gt_image, gt, output_images, output_masks
        _ = gc.collect()


def save_mask(mask_path, mask):
    """
    Save a mask to a file.

    Parameters:
        mask_path (str): The path to save the mask file.
        mask (object): The mask object to be saved.
    """
    with open(mask_path, 'w') as ann:
        json.dump(mask, ann)


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
        mask (dict): The mask object containing annotations and image
        dimensions.

    Returns:
        list: An array of masks converted from the annotations in the
        mask object.
    """
    height, width = mask['images'][0]['height'], mask['images'][0]['width']
    masks = []
    for i in range(len(mask['annotations'])):
        poly_list = get_poly_from_segmentation(mask['annotations'][i]['segmentation'])
        if poly_list is None:
            continue
        cv_mask = poly_to_cv(poly_list, height, width)
        masks.append(cv_mask)
    return masks


def get_bbox_from_poly(poly):
    """
    Calculate the bounding box coordinates from a polygon.

    Parameters:
        poly (list): A list of coordinates representing a polygon. The
            coordinates are in the form of [x1, y1, x2, y2, ..., xn, yn].

    Returns:
        list: A list of four values representing the bounding box coordinates.
            The values are in the form of [min_x, min_y, width, height].
    """
    if len(poly) == 0:
        return [0, 0, 0, 0]
    
    poly_x = [poly[i] for i in range(0, len(poly), 2)]
    poly_y = [poly[i] for i in range(1, len(poly), 2)]
    min_x, min_y = min(poly_x), min(poly_y)
    max_x, max_y = max(poly_x), max(poly_y)
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
    area = 0
    try:
        area = shapely.Polygon(np.array(poly, dtype=np.int8)).area
    except:
        x = [poly[i] for i in range(0, len(poly), 2)]
        y = [poly[i] for i in range(1, len(poly), 2)]
        area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    return area
