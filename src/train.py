"""
This script is used to train a YOLOv8 model on a dataset.
The script is designed to be used in a ClearML environment, where the training process is logged to a ClearML task.

The script may be called according to the following usage information:

    usage: train.py [-h] [--dataset DATASET] [--model MODEL] [--epochs EPOCHS] [--imgsz IMGSZ]

    Train a YOLOv8 model on a dataset

    options:
    -h, --help         show this help message and exit
    --dataset DATASET  Dataset ID to use for training (default: None)
    --model MODEL      Model variant to use for training (default: None)
    --epochs EPOCHS    Number of epochs to train for (default: 10)
    --imgsz IMGSZ      Image size for training (default: 640)
"""

import argparse

required_modules = ['ultralytics', 'clearml']

# Environment Helper functions
def prepare_environment():
    """
    Prepare the environment for training a YOLOv8 model on a dataset.
    Returns a dictionary with the environment setup.

    This dictionary contains the following keys:
    - `PROJECT_NAME`: The name of the ClearML project (Statically set to "LVGL UI Detector")
    - `PROJECT_TAG`: The tag to use for filtering datasets (Statically set to "lvgl-ui-detector")
    - `IN_COLAB`: Whether the environment is running in Google Colab
    - `IN_LOCAL_CONTAINER`: Whether the environment is running in the custom local agent container of the author
    - `DIRS`: A dictionary with all relevant directories for the training process
    - `FILES`: A dictionary with all relevant files for the training process
    - `ENV`: A dictionary with all environment variables

    The dictionary will later be expanded with additional keys as needed.
    """
    import os, sys
    # Set constants for environment
    IN_COLAB = 'google.colab' in sys.modules
    IN_LOCAL_CONTAINER = 'HOMELAB' in os.environ
    env = {
        'PROJECT_NAME': 'LVGL UI Detector',
        'PROJECT_TAG': 'lvgl-ui-detector',
        'IN_COLAB': IN_COLAB,
        'IN_LOCAL_CONTAINER': IN_LOCAL_CONTAINER,
        'DIRS': {},
        'FILES': {},
        'ENV': os.environ,
    }
    # Environment setups
    if IN_COLAB:
        env['DIRS']['root'] = os.path.join("/tmp")
    elif IN_LOCAL_CONTAINER:
        # Local container already comes with the required modules pre-installed
        env['DIRS']['root'] = "/usr/src"
    else:
        # Local development environment needs to have the required modules installed
        env['DIRS']['root'] = os.path.curdir
    # Configure folders
    env['DIRS']['data'] = os.path.join(env['DIRS']['root'], "datasets")
    print(f"Environment setup complete:\nroot: {env['DIRS']['root']}\ndata: {env['DIRS']['data']}")
    return env

def query_datasets(env: dict):
    """
    **Params:**
    - `env` The environment dictionary with the project name and tag.

    Queries the ClearML server for datasets in the project name.

    Stores the dataset information in the environment dictionary, with the dataset ID as the key.
    The provided key,value pairs in the dataset information is provided as-is from the Dataset.list_datasets function, further information is available in the ClearML API documentation.
    """
    from clearml import Dataset
    print(f"Querying datasets for project: {env['PROJECT_NAME']}")
    # datasets = Dataset.list_datasets(dataset_project=env['PROJECT_NAME'], tags=[env['PROJECT_TAG']], only_completed=True)
    datasets = Dataset.list_datasets(dataset_project=env['PROJECT_NAME'])
    # Store dataset filenames per dataset
    env['DATASETS'] = {}
    for dataset in datasets:
        env['DATASETS'][dataset['id']] = dataset
    print(f"Available datasets: {len(datasets)}")

def download_dataset(env: dict, id: str, overwrite: bool = True):
    """
    **Params:**
    - `env` The environment dictionary with the data directory.
    - `id` The ID of the dataset to download.
    - `overwrite` Whether to overwrite the dataset if it already exists. Default is True.
    
    **Returns:**
    - `str` The path to the downloaded dataset.

    Downloads a dataset from ClearML to the local environment.

    The dataset is downloaded to the "data" directory of the environment dictionary.
    The dataset is downloaded as a mutable copy, as this is required for the dataset to be locally available for the training process.
    No actual modifications are made to the dataset, as the dataset is only used for training purposes.
    """
    from clearml import Dataset
    print(f"Downloading dataset: {id}")
    dataset = Dataset.get(id)
    return dataset.get_mutable_local_copy(env['DIRS']['data'], overwrite=overwrite)

def fix_dataset_path(file: str, replacement_path: str):
    """
    **Params:**
    - `file` The path to the dataset YAML file.
    - `replacement_path` The path to replace the dataset path with.
    
    **Returns:**
    - `dict` The modified contents of the dataset YAML file.

    Fixes the dataset path in a dataset YAML file.

    Since the dataset is downloaded to a different location than the original, the path in the dataset YAML file needs to be adjusted.
    This function reads the dataset YAML file, adjusts the path to where the dataset was downloaded and writes the adjusted dataset back to the file.
    The modified contents of the dataset YAML file are returned.
    """
    import yaml
    print(f"Adjusting dataset path: {file} -> {replacement_path}")
    # Replace path in dataset file to match current environment
    with open(file, 'r+') as f:
        dataset_content = yaml.safe_load(f)
        dataset_content['path'] = replacement_path
        print(f"Original dataset:\n{dataset_content}")
        f.seek(0)
        yaml.dump(dataset_content, f)
        f.truncate()
        f.seek(0)
        print(f"Adjusted dataset:\n{f.read()}")
        return dataset_content

# Training helper functions
def prepare_training(env: dict, model_variant: str, dataset_id: str, args: dict, project: str = "LVGL UI Detector"):
    """
    **Params:**
    - `env` The environment dictionary with the dataset information.
    - `model_variant` The model variant to train.
    - `dataset_id` The ID of the dataset to use for training.
    - `args` The arguments to use for training.
    - `project` The name of the ClearML project to use for training. Default is "LVGL UI Detector".
    
    **Returns:**
    - `str` The ID of the created ClearML task.

    Prepares the traiining environment for the YOLO (ultralytics engine) training process.

    It downloads the dataset, adjusts the dataset path in the dataset YAML file and creates a ClearML task for the training process.
    The created task ID is returned for further usage and reference.
    """
    from clearml import Task, Dataset
    import os
    print(f"Preparing {model_variant} for dataset: {dataset_id}")
    env['ID'] = dataset_id
    # Fetch dataset YAML
    env['FILES'][dataset_id] = Dataset.get(dataset_id).list_files("*.yaml")
    print(env['FILES'][dataset_id])
    # Download & modify dataset
    env['DIRS']['target'] = download_dataset(env, dataset_id)
    dataset_file = os.path.join(env['DIRS']['target'], env['FILES'][dataset_id][0])
    dataset_content = fix_dataset_path(dataset_file, env['DIRS']['target'])
    args['data'] = os.path.join(env['DIRS']['target'], env['FILES'][dataset_id][0])
    # Create a ClearML Task
    task = Task.init(
        project_name="LVGL UI Detector",
        task_name=f"Train {model_variant} ({env['DATASETS'][dataset_id]['name']})",
        task_type=Task.TaskTypes.training
    )
    task.connect(args)
    # Log "model_variant" parameter to task
    task.set_parameter("model_variant", model_variant)
    task.set_parameter("dataset", dataset_id)
    # task.set_parameter("General/data", args['data'])
    task.connect_configuration(name="Dataset YAML", configuration=args['data'])
    task.connect_configuration(name="Dataset Content", configuration=dataset_content)
    return task.id

def training_task(env: dict, model_variant: str, args: dict):
    """
    **Params:**
    - `env` The environment dictionary with the dataset information.
    - `model_variant` The model variant to train.
    - `args` The arguments to use for training.
    
    **Returns:**
    - `dict` The results of the training process.
    - `str` The ID of the ClearML task.

    Trains a YOLO model using the ultralytics engine, based on the provided arguments and model_variant.

    Through the ultralytics engine, the training process is automatically logged to the existing ClearML task.
    The results of the training process are returned, along with the task ID for reference.
    """
    from ultralytics import YOLO
    from clearml import Task
    import os
    task = Task.current_task()
    # Load the YOLOv8 model
    model = YOLO(f'{model_variant}.pt')
    if not args['data'].startswith(env['DIRS']['target']):
        print(f"Dataset path mismatch: {args['data']} -> {os.path.join(env['DIRS']['target'], env['FILES'][env['ID']][0])}")
        args['data'] = os.path.join(env['DIRS']['target'], env['FILES'][env['ID']][0])
        task.set_parameter("General/data", args['data'])
    print(f"Training {model_variant} on dataset: {args['data']}")
    # Train the model 
    # If running remotely, the arguments may be overridden by ClearML if they were changed in the UI
    try:
        results = model.train(**args)
    except Exception as e:
        raise e
    finally:
        task.close()
    return results, task.id

if __name__ == "__main__":
    from clearml import Task
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model on a dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, help="Dataset ID to use for training")
    parser.add_argument("--model", type=str, help="Model variant to use for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    args = parser.parse_args()
    env = prepare_environment()
    query_datasets(env)
    if args.dataset not in env['DATASETS'].keys():
        print("Dataset ID not found.")
    dataset_choice = args.dataset
    # Training inputs (initial short training run to get a task ID for optimization)
    model_variant = args.model
    varargs = dict(
        epochs=args.epochs, 
        imgsz=args.imgsz
    )
    id = prepare_training(env, model_variant, dataset_choice, varargs)
    results = training_task(env, model_variant, varargs)
    print(f"Training task ID: {id}")
