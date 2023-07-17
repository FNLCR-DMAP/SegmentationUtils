import os, math, json
import numpy as np
import pandas as pd


def validate_split(trainf,valf,testf,e=1e-5):
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
    assert fraction_sum < 1+e and fraction_sum > 1-e, \
        f"""Split fractions dont sum up to 1 (100%), please check fractions:
        Train: {trainf}
        Validation: {valf}
        Test: {testf}"""


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
        assert i_test == size, "The sum over split sets is not the total number of elements."

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


def create_annotations(input_folder,output_file,ids=[],annotation_suffix="_coco.json"):
    """
    Creates annotations for a given input folder and writes them to an output file.
    
    Parameters:
        input_folder (str): The path to the folder containing the input files.
        output_file (str): The path to the output file where the annotations will be written.
        ids (list, optional): A list of specific ids to include in the annotations. Defaults to an empty list.
        annotation_suffix (str, optional): The suffix for the annotation files. Defaults to "_coco.json".
        
    Returns:
        dict: The merged data containing the images, annotations, and categories.
    """
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    annotation_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    if len(ids) > 0:
        fann = []
        to_rm = len(annotation_suffix)
        for a in annotation_files:
            if a[:-to_rm] in ids:
                fann.append(a)
        annotation_files = fann

    annotation_id = 1
    image_id = 1

    for file in annotation_files:
        with open(os.path.join(input_folder, file), 'r') as f:
            data = json.load(f)

        if not merged_data["categories"]:
            merged_data["categories"] = data["categories"]

        for image in data["images"]:
            image["id"] = image_id
            merged_data["images"].append(image)
            image_id += 1

        for annotation in data["annotations"]:
            annotation["id"] = annotation_id
            annotation["image_id"] = annotation["image_id"] + (image_id - 1)
            annotation_id += 1
            merged_data["annotations"].append(annotation)

    print(f"Created annotation file with {len(annotation_files)} images.")
    with open(output_file, 'w') as f:
        json.dump(merged_data, f)
    
    return merged_data


def create_split_annotations(train_ids,val_ids,test_ids,annotations_path,output_path,annotation_suffix):
    """
    Create split annotations.

    Parameters:
        train_ids (list): List of training ids.
        val_ids (list): List of validation ids.
        test_ids (list): List of testing ids.
        annotations_path (str): Path to the annotations file.
        output_path (str): Path to the output directory.
        annotation_suffix (str): Annotation suffix.

    Returns:
        tuple: A tuple containing the split configuration and the merged annotations for training, validation, and testing.
    """
    # Print split sizes
    print(f" > Split sizes:")
    print(f"Training split size:        {len(train_ids)}")
    print(f"Validation split size:      {len(val_ids)}")
    print(f"Testing split size:         {len(test_ids)}")

    # Save split into a json file
    config = {'train_ids':train_ids, 'val_ids':val_ids, 'test_ids':test_ids}
    with open(f"{output_path}/split.json", "w") as outfile:
        json.dump(config, outfile, indent=4)
    
    # Create the merged annotation file for each set
    train_ann = create_annotations(annotations_path, f"{output_path}/train_annotations.json",      train_ids, annotation_suffix)
    val_ann   = create_annotations(annotations_path, f"{output_path}/validation_annotations.json", val_ids,   annotation_suffix)
    test_ann  = create_annotations(annotations_path, f"{output_path}/test_annotations.json",       test_ids,  annotation_suffix)

    # Print merged size
    print(f" > Annotation sizes:")
    print(f"Annotations for training:   {len(train_ann['images'])}")
    print(f"Annotations for validation: {len(val_ann['images'])}")
    print(f"Annotations for testing:    {len(test_ann['images'])}")

    # Assert that split size and merged size are the same
    assert len(train_ids) == len(train_ann['images']), "Expected number of ids for training is different than what was obtained in merging annotation files, please check."
    assert len(val_ids)   == len(val_ann['images']),   "Expected number of ids for validation is different than what was obtained in merging annotation files, please check."
    assert len(test_ids)  == len(test_ann['images']),  "Expected number of ids for testing is different than what was obtained in merging annotation files, please check."
    
    return config, train_ann, val_ann, test_ann


def augment_ids(train_ids,val_ids,test_ids,augmented_path,annotation_suffix,aug_train_only=True):
    """
    Augments the given IDs with additional IDs from the augmented_path directory.
    
    Parameters:
        train_ids (list): The list of train IDs.
        val_ids (list): The list of validation IDs.
        test_ids (list): The list of test IDs.
        augmented_path (str): The path to the directory containing augmented files.
        annotation_suffix (str): The suffix of annotated files.
    
    Returns:
        tuple: A tuple containing the augmented train IDs, augmented validation IDs, 
               and augmented test IDs.
    """
    aug_train_ids, aug_val_ids, aug_test_ids = [], [], []
    augmented_files = [f[:-len(annotation_suffix)] for f in os.listdir(augmented_path) if f.endswith(annotation_suffix)]
    for tid in train_ids:
        to_include = [a for a in augmented_files if a.startswith(tid)]
        if len(to_include) > 0:
            aug_train_ids.extend(to_include)
    
    for vid in val_ids:
        to_include = [a for a in augmented_files if a.startswith(vid)]
        if len(to_include) > 0:
            if aug_train_only:
                to_include.sort()
                aug_val_ids.append(to_include[-1])
            else:
                aug_val_ids.extend(to_include)
    for tid in test_ids:
        to_include = [a for a in augmented_files if a.startswith(tid)]
        if len(to_include) > 0:
            if aug_train_only:
                to_include.sort()
                aug_test_ids.append(to_include[-1])
            else:
                aug_test_ids.extend(to_include)
    return aug_train_ids, aug_val_ids, aug_test_ids


def create_split(
        annotations_path,
        output_path         = "output",
        annotation_suffix   = "_coco.json",
        train_fraction      = 0.7,
        validation_fraction = 0.2,
        test_fraction       = 0.1,
        augmented_path      = None):
    """
    Create a train/validation/test split of annotation files.

    Parameters:
        annotations_path (str): The path to the directory containing the annotation files.
        output_path (str, optional): The output path for the split annotations. Defaults to "output".
        annotation_suffix (str, optional): The suffix of the annotation files. Defaults to "_coco.json".
        train_fraction (float, optional): The fraction of data to be included in the training set. Defaults to 0.7.
        validation_fraction (float, optional): The fraction of data to be included in the validation set. Defaults to 0.2.
        test_fraction (float, optional): The fraction of data to be included in the test set. Defaults to 0.1.
        augmented_path (str): The path to the directory containing the augmented annotation files. Defaults to None.

    Returns:
        The annotations for the split, based on the specified fractions.
    """
    # Validate fractions
    validate_split(train_fraction,validation_fraction,test_fraction)

    # Get annotation file ids
    arr = [f for f in os.listdir(annotations_path) if annotation_suffix in f]
    size = len(arr)
    to_rm = len(annotation_suffix)
    for i in range(len(arr)):
        arr[i] = arr[i][:-to_rm]

    # Shuffle the array
    np.random.shuffle(arr)

    # Split the array
    train_ids, val_ids, test_ids = split_array(arr,size,validation_fraction,test_fraction)

    # Create the annotations based on the split
    if augmented_path is not None:
        train_ids,val_ids,test_ids = augment_ids(train_ids,val_ids,test_ids,augmented_path,annotation_suffix)
        annotations_path = augmented_path
    return create_split_annotations(train_ids,val_ids,test_ids,annotations_path,output_path,annotation_suffix)


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
        augmented_path      = None):
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
        augmented_path (str): The path to the directory containing the augmented annotation files. Defaults to None.
    
    Returns:
        str: Path to the created split annotations file.
    """
    # Validate fractions
    validate_split(train_fraction,validation_fraction,test_fraction)

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

        # Shuffle the array
        np.random.shuffle(arr)

        # Split the array into groups
        train_s, val_s, test_s = split_array(arr,size,validation_fraction,test_fraction)
        train_ids.extend(train_s)
        val_ids.extend(val_s)
        test_ids.extend(test_s)

    # Create the annotations based on the split
    if augmented_path is not None:
        train_ids,val_ids,test_ids = augment_ids(train_ids,val_ids,test_ids,augmented_path,annotation_suffix)
        annotations_path = augmented_path
    return create_split_annotations(train_ids,val_ids,test_ids,annotations_path,output_path,annotation_suffix)


def get_split_ids(f):
    """
    Reads a JSON file and extracts the 'id' and 'file_name' values from each image object.

    Parameters:
        f (str): The path to the JSON file.

    Returns:
        list: A list of tuples containing the 'id' and 'file_name' values for each image.
    """
    with open(f,'r') as file:
        ann = json.load(file)
    data = [(a['id'],a['file_name']) for a in ann['images']]
    return data