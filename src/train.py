import argparse
required_modules = ['ultralytics', 'clearml']

# Environment Helper functions
def prepare_environment():
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
    from clearml import Dataset
    print(f"Querying datasets for project: {env['PROJECT_NAME']}")
    datasets = Dataset.list_datasets(env['PROJECT_NAME'], partial_name="UI Randomizer", tags=[env['PROJECT_TAG']], only_completed=True)
    # Store dataset filenames per dataset
    env['DATASETS'] = {}
    for dataset in datasets:
        env['DATASETS'][dataset['id']] = dataset
    print(f"Available datasets: {len(datasets)}")

def download_dataset(env: dict, id: str, overwrite: bool = True):
    from clearml import Dataset
    print(f"Downloading dataset: {id}")
    dataset = Dataset.get(id)
    return dataset.get_mutable_local_copy(env['DIRS']['data'], overwrite=overwrite)

def fix_dataset_path(file: str, replacement_path: str):
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
    from ultralytics import YOLO
    from clearml import Task
    import os
    task = Task.current_task()
    # Load the YOLOv8 model
    model = YOLO(f'{model_variant}.pt')
    if not args['data'].startswith(env['DIRS']['target']):
        print(f"Dataset path mismatch: {args['data']} -> {os.path.join(env['DIRS']['target'], env['FILES'][env['ID']][0])}")
        args['data'] = os.path.join(env['DIRS']['target'], env['FILES'][env['ID']][0])
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
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model on a dataset")
    parser.add_argument("--dataset", type=str, help="Dataset ID to use for training")
    parser.add_argument("--model", type=str, help="Model variant to use for training")
    args = parser.parse_args()
    env = prepare_environment()
    query_datasets(env)
    if args.dataset not in env['DATASETS'].keys():
        print("Dataset ID not found.")
    dataset_choice = args.dataset
    # Training inputs (initial short training run to get a task ID for optimization)
    model_variant = args.model
    varargs = dict(
        epochs=3, 
        imgsz=480
    )
    id = prepare_training(env, model_variant, dataset_choice, varargs)
    results = training_task(env, model_variant, varargs)
    print(f"Training task ID: {id}")
