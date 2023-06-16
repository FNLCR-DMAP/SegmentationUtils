import os, math, json
import numpy as np
import pandas as pd

def coco_conversion(input_path,output_path):
    # TO DO
    return


def validate_split(trainf,valf,testf,e=1e-5):
    """
    The validate_split function is used to ensure that the sum of train, validation, and test fractions
    is 1 within a given tolerance e. This helps guarantee that the proportions of the dataset are properly
    allocated between the three sets. Checks whether it falls within the provided epsilon range
    (i.e., between 1-e and 1+e). If the condition fails, an AssertionError is raised.
    
    Parameters
    ----------
    trainf : float
        Train fraction.

    valf : float
        Validation fraction.

    testf : float
        Test fraction.

    e : float
        Tolerance
    """
        
    fraction_sum = trainf + valf + testf
    assert fraction_sum < 1+e and fraction_sum > 1-e, \
        f"""Split fractions dont sum up to 1 (100%), please check fractions:
        Train: {trainf}
        Validation: {valf}
        Test: {testf}"""


def split_array(arr,size,vf,tf):
    """
    This function splits the given array of image IDs into training, validation and testing subsets while trying
    to maintain the percentages of each set.
    
    Parameters
    ----------
    arr : list(string)
        Array containing the image IDs.

    size : int
        Size of the array.

    vf : float
        Validation set fraction.

    tf : float
        Test set fraction.
    
    Returns
    -------
    train : list(string)
        Array containing the image IDs for the training set.

    val : list(string)
        Array containing the image IDs for the validation set.

    test : list(string)
        Array containing the image IDs for the test set.

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
    This function use the list of id files to read the coco format annotations of these ids and save
    into a merged annotation file.
    
    Parameters
    ----------
    input_folder : string
        Folder path containing annotation files in coco format.

    output_file : string
        Name of the output file to save the merged annotations.

    ids : list(string)
        List of the image ids to include in the merged annotations.

    annotation_suffix : string
        In case the coco format annotations has a suffix after the name of the image (ID).
    
    Returns
    -------
    merged_data : dictionary
        Coco format annotations containing all of the images and their annotations.
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

    with open(output_file, 'w') as f:
        json.dump(merged_data, f)
    
    return merged_data


def create_split_annotations(train_ids,val_ids,test_ids,annotations_path,output_path,annotation_suffix):
    """
    This function receives the image IDs (names) of each of the split sets and returns the merged coco format
    annotation files for each of the sets.
    
    Parameters
    ----------
    train_ids : list(string)
        Array containing the list of image IDs for the training set.

    val_ids : list(string)
        Array containing the list of image IDs for the validation set.

    test_ids : list(string)
        Array containing the list of image IDs for the test set.

    annotations_path : string
        Folder path containing the annotation files in coco format.

    output_path : string
        Folder path where the dictionaries of the merged annotations will be saved.

    annotation_suffix : string
        In case the coco format annotations has a suffix after the name of the image (ID).
    
    Returns
    -------
    config : dictionary
        Dictionary containing the image IDs for each set.

    train_ann : dictionary
        Coco format annotations for all of the training images and their annotations.

    val_ann : dictionary
        Coco format annotations for all of the validation images and their annotations.

    test_ann : dictionary
        Coco format annotations for all of the testing images and their annotations.
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

def create_split(
        annotations_path,
        output_path         = "output",
        annotation_suffix   = "_coco.json",
        train_fraction      = 0.7,
        validation_fraction = 0.2,
        test_fraction       = 0.1):
    """
    This function creates the split of a set of images based on a folder containing the annotations,
    the suffix of the annotation file names, the output path and the fractions of the split. It saves
    the coco format annotation dictionaries as a JSON file inside the output path but also return them
    for the user as dictionaries.
    
    Parameters
    ----------
    annotations_path : string
        Folder path containing the annotation files in coco format.

    output_path : string
        Folder path where the dictionaries of the merged annotations will be saved.

    annotation_suffix : string
        In case the coco format annotations has a suffix after the name of the image (ID).

    train_fraction : float
        Train set fraction.

    validation_fraction : float
        Validation set fraction.

    test_fraction : float
        Test set fraction.
    
    Returns
    -------
    create_split_annotations : function
        Returns the merged coco format annotation files (dictionary) for each of the sets.
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
        test_fraction       = 0.1):
    """
    This function creates the split of a set of images based on a csv file containing the clustering
    analysis labels for all of the image IDs to be used. It saves the coco format annotation dictionaries
    as a JSON file inside the output path but also return them for the user as dictionaries.
    
    Parameters
    ----------
    cluster_file : string
        File path for the csv file containing the clustering labels and image IDs.

    cluster_column : string
        Column name on the csv file containing the clustering labels.

    image_column : string
        Column name on the csv file containing the image IDs.

    annotations_path : string
        Folder path containing the annotation files in coco format.

    output_path : string
        Folder path where the dictionaries of the merged annotations will be saved.

    annotation_suffix : string
        In case the coco format annotations has a suffix after the name of the image (ID).

    train_fraction : float
        Train set fraction.

    validation_fraction : float
        Validation set fraction.

    test_fraction : float
        Test set fraction.
    
    Returns
    -------
    create_split_annotations : function
        Returns the merged coco format annotation files (dictionary) for each of the sets.
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
    return create_split_annotations(train_ids,val_ids,test_ids,annotations_path,output_path,annotation_suffix)


def get_split_ids(f):
    """
    This function extract specific pieces of information from the given JSON file "f". Specifically,
    the script focuses on retrieving two pieces of data - the 'ID' field and 'filename' field
    associated with every item listed under the "Images" key in the main JSON structure.
    
    Parameters
    ----------
    f : string
        The JSON file name.

    Returns
    -------
    data : array of string tuples
        The extracted image id and file name for each image item within the file.
    """
    with open(f,'r') as file:
        ann = json.load(file)
    data = [(a['id'],a['file_name']) for a in ann['images']]
    return data