import argparse

required_modules = ['ultralytics', 'clearml']

def prepare_task(args: dict):
    from clearml import Task
    from clearml import Dataset
    dataset = Dataset.get(args.dataset)
    dataset_id = dataset.id
    dataset_name = dataset.name
    dataset_tags = dataset.tags
    task = Task.init(
        project_name="LVGL UI Detector", 
        task_name=f"Tune {args.model} ({dataset_name})", 
        task_type=Task.TaskTypes.optimizer,
        tags=['tune', args.model, *dataset_tags])
    varargs = dict(
        epochs=args.epochs, 
        iterations=args.iterations,
        optimizer=args.optimizer,
        imgsz=args.imgsz
    )
    task.connect_configuration(name="Tune/Args", configuration=varargs, description="Tuning arguments")
    task.set_parameter("Tune/model_variant", model_variant)
    task.set_parameter("Tune/dataset", dataset_id)
    return varargs

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
    # datasets = Dataset.list_datasets(dataset_project=env['PROJECT_NAME'], tags=[env['PROJECT_TAG']], only_completed=True)
    datasets = Dataset.list_datasets(dataset_project=env['PROJECT_NAME'])
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

# Tuning helper functions
def prepare_tuning(env: dict, model_variant: str, dataset_id: str, args: dict, project: str = "LVGL UI Detector"):
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

def yolo_tune(env: dict, model_variant: str, args: dict, store_task: bool = False):
    from ultralytics import YOLO
    from clearml import Task
    import os
    # Initialize the YOLO model
    model = YOLO(f'{model_variant}.pt')
    if store_task:
        task = Task.current_task()
    # task = Task.current_task()
    if not args['data'].startswith(env['DIRS']['target']):
        print(f"Dataset path mismatch: {args['data']} -> {os.path.join(env['DIRS']['target'], env['FILES'][env['ID']][0])}")
        args['data'] = os.path.join(env['DIRS']['target'], env['FILES'][env['ID']][0])
        if store_task:
            task.set_parameter("General/data", args['data'])
    try:
        results = model.tune(**args) # Individual results are stored via automatic integration
    except Exception as e:
        raise e
    finally:
        print(results)
    return results, task.id if store_task else None


if __name__ == "__main__":
    # args: model: str, data: str, epochs: int, iterations: int, optimizer: str
    import argparse
    parser = argparse.ArgumentParser(description='Tune dataset hyperparameters for a YOLO model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, help=f'Model variant to use for tuning')
    parser.add_argument('--dataset', type=str, help='Dataset ID to use for tuning')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations to tune for')
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for tuning")
    parser.add_argument('--optimizer', type=str, default='AdamW', help='YOLO Optimizer (from: Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto)')
    parser.add_argument('--store_task', action='store_true', default=False, help='Store the tuning process as a ClearML task (individual results will not be stored as seperate tasks)')
    args = parser.parse_args()
    if args.store_task:
        print("Storing & running tuning task.")
        prepare_task(args)
    else:
        print("Starting tuner.")
        varargs = dict(
            epochs=args.epochs, 
            iterations=args.iterations,
            optimizer=args.optimizer,
            imgsz=args.imgsz
        )
    env = prepare_environment()
    query_datasets(env)
    if args.dataset not in env['DATASETS'].keys():
        print("Dataset ID not found.")
    dataset_choice = args.dataset
    model_variant = args.model
    prepare_tuning(env, model_variant, dataset_choice, varargs)
    results, id = yolo_tune(env, model_variant, varargs, store_task=args.store_task)
    print(f"Tune task ID: {id}")
    print("Tuning completed.")
