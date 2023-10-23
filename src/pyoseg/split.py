import os, math, json
import numpy as np
import pandas as pd
from pyoseg.augmentation import augment_data

TEST_MODE = False

def validate_split(trainf, valf, testf, e=1e-5):
    """
    Validate the split fractions.
    
    Parameters:
        trainf (float): The fraction of the training data.
        valf (float): The fraction of the validation data.
        testf (float): The fraction of the test data.
        e (float, optional): A small epsilon value to account for floating point errors. Defaults to 1e-5.
    
    Raises:
        AssertionError: If the sum of the fractions is not approximately equal to 1.
    
    Returns:
        None
    """
    fraction_sum = trainf + valf + testf
    if not (fraction_sum < 1+e and fraction_sum > 1-e):
        raise ValueError(f"""Split fractions dont sum up to 1 (100%), please check fractions:\nTrain: {trainf}\nValidation: {valf}\nTest: {testf}""")


def split_array(arr,size,vf,tf):
    """
    Split an array into training, validation, and testing sets.

    Parameters:
        arr (list): The input array to be split.
        size (int): The size of the input array.
        vf (float): The validation set fraction.
        tf (float): The test set fraction.

    Returns:
        train (list): The training set.
        val (list): The validation set.
        test (list): The testing set.
    """
    if size == 0:
        print(f"WARNING: Empty array.")
        return [],[],[]
    if size == 1:
        return [arr[0]],[],[]
    if size == 2:
        return [arr[0]],[arr[1]],[]
    if size > 2:
        n_test = math.ceil(size*tf)
        n_val = math.ceil(size*vf)
        n_train = size - n_test - n_val
        i_val = n_train + n_val
        i_test = i_val + n_test
        if i_test != size:
            raise ValueError(f"The sum over split sets is not the total number of elements.")

        train = arr[:n_train]
        val = arr[n_train:i_val]
        test = arr[i_val:i_test]

        if n_train == 1:
            train = list(train)
        if n_val == 1:
            val = list(val)
        if n_test == 1:
            test = list(test)
        return train, val, test


def create_annotations(input_folder, output_file, ids=None, annotation_suffix="_coco.json", random=False):
    """
    Creates annotations for a given input folder and writes them to an output file.
    
    Parameters:
        input_folder (str): The path to the folder containing the input files.
        output_file (str): The path to the output file where the annotations will be written.
        ids (list, optional): A list of specific ids to include in the annotations. Defaults to None.
        annotation_suffix (str, optional): The suffix for the annotation files. Defaults to "_coco.json".
        random (bool, optional): Whether to create in random order. Defaults to False.
        
    Returns:
        dict: The merged data containing the images, annotations, and categories.
    """
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    annotation_files = [f for f in os.listdir(input_folder) if f.endswith(annotation_suffix)]
    if ids is not None:
        fann = []
        to_rm = len(annotation_suffix)
        for a in annotation_files:
            if a[:-to_rm] in ids:
                fann.append(a)
        annotation_files = fann

    if ids is None or len(ids) != 0:
        if len(annotation_files) == 0:
            raise FileNotFoundError(f"No annotation files found in input folder {input_folder}.")
        if random:
            annotation_files = np.random.permutation(annotation_files)
        
        if TEST_MODE:
            annotation_files.sort()

        image_id = 0
        annotation_id = 0
        for file in annotation_files:
            with open(os.path.join(input_folder, file), 'r') as f:
                data = json.load(f)

            if not merged_data["categories"]:
                merged_data["categories"] = data["categories"]

            if len(data["images"]) > 1:
                raise ValueError("Expected only one image per file.")
            
            image = data["images"][0]
            image["id"] = image_id
            merged_data["images"].append(image)

            for annotation in data["annotations"]:
                annotation["image_id"] = image_id
                annotation["id"] = annotation_id
                merged_data["annotations"].append(annotation)
                annotation_id += 1
            
            image_id += 1

    print(f"Creating annotation file with {len(annotation_files)} images...")
    with open(output_file, 'w') as f:
        json.dump(merged_data, f)
    print(f"File created: {output_file}")
    
    return merged_data


def create_split_annotations(train_ids, val_ids, test_ids, annotations_path, output_path, annotation_suffix, augmentation=None):
    """
    Create split annotations.

    Parameters:
        train_ids (list): List of training ids.
        val_ids (list): List of validation ids.
        test_ids (list): List of testing ids.
        annotations_path (str): Path to the annotations file.
        output_path (str): Path to the output directory.
        annotation_suffix (str): Annotation suffix.
        augmentation (dict): Configuration on how to augment the data.

    Returns:
        tuple: A tuple containing the split configuration and the merged annotations for training, validation, and testing.
    """
    # Print split sizes

    # Save split into a json file
    config = {'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids}
    with open(f"{output_path}/split.json", "w") as outfile:
        json.dump(config, outfile, indent=4)

    train_path, val_path, test_path = annotations_path, annotations_path, annotations_path
    if augmentation is not None:
        ids, paths = augment_ids(
            config, annotations_path, output_path, annotation_suffix, augmentation, shuffle=True)
        train_ids, val_ids, test_ids = ids["train_ids"], ids["val_ids"], ids["test_ids"]
        train_path, val_path, test_path = paths["train_ids"], paths["val_ids"], paths["test_ids"]
    
    # Create the merged annotation file for each set
    print(f" > Split sizes:")
    print(f"Training split size:         {len(train_ids)}")
    train_ann = create_annotations(
        train_path, f"{output_path}/train_annotations.json", train_ids, annotation_suffix)
    print(f"Validation split size:       {len(val_ids)}")
    val_ann = create_annotations(
        val_path, f"{output_path}/validation_annotations.json", val_ids, annotation_suffix)
    print(f"Testing split size:          {len(test_ids)}")
    test_ann = create_annotations(
        test_path, f"{output_path}/test_annotations.json", test_ids, annotation_suffix)

    # Assert that split size and merged size are the same
    if len(train_ids) != len(train_ann['images']) or len(val_ids) != len(val_ann['images']) or len(test_ids) != len(test_ann['images']):
        raise ValueError("Expected number of ids for training, validation and testing is different than what was obtained while merging annotation files.")
    
    return config, train_ann, val_ann, test_ann


def augment_ids(split, annotations_path, output_path, annotation_suffix, augmentation, shuffle=True):
    """
    Generates a dictionary of augmented image ids and paths for each split.

    Parameters:
        split (dict): A dictionary containing the splits as keys and a list of file names as values.
        annotations_path (str): The path to the directory containing the annotations.
        output_path (str): The path to the directory where the augmented images will be saved.
        annotation_suffix (str): The suffix of the annotation files.
        augmentation (dict): A dictionary containing the augmentation configuration.
        shuffle (bool, optional): Whether to shuffle the augmented image ids. Defaults to True.

    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary contains the augmented image ids for
            each split, and the second dictionary contains the corresponding paths to the augmented images.
    """

    dir_data = augmentation["input_path"]
    dir_anns = annotations_path
    ids = {}
    paths = {}

    for key in split.keys():
        os.makedirs(f"{output_path}/{key}", exist_ok=True)
        os.makedirs(f"{output_path}/{key}/data", exist_ok=True)
        os.makedirs(f"{output_path}/{key}/annotations", exist_ok=True)
        for file in split[key]:
            os.system(f"cp {dir_data}/{file}.png {output_path}/{key}/data")
            os.system(f"cp {dir_anns}/{file}_coco.json {output_path}/{key}/annotations")

        i_path = f"{output_path}/{key}/augmented"
        paths[key] = i_path
        os.makedirs(i_path, exist_ok=True)
        augment_data(
            data_path = f"{output_path}/{key}/data",
            annotations_path = f"{output_path}/{key}/annotations",
            output_path = i_path,
            functions = augmentation[key.split('_')[0]]["functions"],
            times = augmentation[key.split('_')[0]]["times"])
        i_id = [f[:-len(annotation_suffix)] for f in os.listdir(f"{output_path}/{key}/augmented") if annotation_suffix in f]
        if shuffle:
            i_id = np.random.permutation(i_id)
        ids[key] = i_id

    return ids, paths


def create_split(
        annotations_path,
        output_path         = "output",
        annotation_suffix   = "_coco.json",
        train_fraction      = 0.7,
        validation_fraction = 0.2,
        test_fraction       = 0.1,
        seed                = None,
        augmentation        = None):
    """
    Create a train/validation/test split of annotation files.

    Parameters:
        annotations_path (str): The path to the directory containing the annotation files.
        output_path (str, optional): The output path for the split annotations. Defaults to "output".
        annotation_suffix (str, optional): The suffix of the annotation files. Defaults to "_coco.json".
        train_fraction (float, optional): The fraction of data to be included in the training set. Defaults to 0.7.
        validation_fraction (float, optional): The fraction of data to be included in the validation set. Defaults to 0.2.
        test_fraction (float, optional): The fraction of data to be included in the test set. Defaults to 0.1.
        augmentation (dict): Configuration on how to augment the data. Defaults to None.
        seed (int): The seed for the random number generator. Defaults to None.

    Returns:
        The annotations for the split, based on the specified fractions.
    """
    # Validate fractions
    validate_split(train_fraction, validation_fraction, test_fraction)

    # Get annotation file ids
    arr = [f for f in os.listdir(annotations_path) if annotation_suffix in f]
    size = len(arr)
    to_rm = len(annotation_suffix)
    for i in range(len(arr)):
        arr[i] = arr[i][:-to_rm]
    
    if TEST_MODE and size > 0:
        arr.sort()

    # Shuffle the array
    if seed is not None and seed >= 0:
        np.random.seed(seed)
        np.random.shuffle(arr)

    # Split the array
    train_ids, val_ids, test_ids = split_array(arr,size, validation_fraction, test_fraction)

    # Create the annotations based on the split
    return create_split_annotations(train_ids, val_ids, test_ids, annotations_path, output_path, annotation_suffix, augmentation)


def create_split_cluster(
        cluster_file        = "Membrane_small_img_clusters.csv",
        cluster_column      = "PhenoGraph_clusters",
        image_column        = "Images",
        annotations_path    = "",
        output_path         = "output",
        annotation_suffix   = "_coco.json",
        train_fraction      = 0.7,
        validation_fraction = 0.2,
        test_fraction       = 0.1,
        seed                = None,
        augmentation        = None):
    """
    Create split clusters based on a clustering file and annotation images.
    
    Parameters:
        cluster_file (str): Path to the clustering file (default is "Membrane_small_img_clusters.csv").
        cluster_column (str): Name of the column in the clustering file that contains the cluster labels (default is "PhenoGraph_clusters").
        image_column (str): Name of the column in the clustering file that contains the image filenames (default is "Images").
        annotations_path (str): Path to the directory containing the annotation images.
        output_path (str): Path to the directory where the split clusters will be saved (default is "output").
        annotation_suffix (str): Suffix of the annotation image files (default is "_coco.json").
        train_fraction (float): Fraction of the data to use for training (default is 0.7).
        validation_fraction (float): Fraction of the data to use for validation (default is 0.2).
        test_fraction (float): Fraction of the data to use for testing (default is 0.1).
        seed (int): The seed for the random number generator. Defaults to None.
        augmentation (str): The configuration on how to augment the data. Defaults to None.
    
    Returns:
        str: Path to the created split annotations file.
    """
    # Validate fractions
    validate_split(train_fraction, validation_fraction, test_fraction)

    # Get annotation file ids
    current_imgs = np.array([f for f in os.listdir(annotations_path) if annotation_suffix in f])
    to_rm = len(annotation_suffix)
    for i in range(len(current_imgs)):
        current_imgs[i] = current_imgs[i][:-to_rm] + ".png"
    
    # Read clustering file and extract clusters
    df = pd.read_csv(cluster_file)
    clusters = df[cluster_column].unique()
    clusters.sort()

    # Initialize arrays
    train_ids, val_ids, test_ids = [], [], []

    # Loop over clusters
    for c in clusters:
        # Extract cluster data as an array, filtered by the images we have available 
        cdf = df[df[cluster_column] == c]
        arr = np.array(cdf[cdf[image_column].isin(current_imgs)][image_column].to_list())
        size = len(arr)
        print(f"Size of cluster '{c}': {size}")
        if size == 0:
            continue

        for i in range(size):
            arr[i] = arr[i].split(".")[0]

        if TEST_MODE:
            arr.sort()
            
        # Shuffle the array
        if seed is not None and seed >= 0:
            np.random.seed(seed)
            np.random.shuffle(arr)

        # Split the array into groups
        train_s, val_s, test_s = split_array(arr, size, validation_fraction, test_fraction)
        train_ids.extend(train_s)
        val_ids.extend(val_s)
        test_ids.extend(test_s)

    # Create the annotations based on the split
    return create_split_annotations(train_ids, val_ids, test_ids, annotations_path, output_path, annotation_suffix, augmentation)
