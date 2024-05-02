import argparse
from importlib.metadata import metadata
from re import split

implemented_types = ["arc", 
                     "bar", "button", "buttonmatrix", 
                     "calendar", "checkbox", 
                     "dropdown", 
                     "label", "led", 
                     "roller", 
                     "scale", "slider", "spinbox", "switch", 
                     "table", "textarea"]

# Generators
def capture_random(env: dict, args: argparse.Namespace):
    import subprocess, os, shutil, glob
    from clearml import Task
    import numpy as np
    task = Task.current_task()
    logger = task.get_logger()
    global implemented_types
    widgets = {}
    for widget in implemented_types:
        widgets[widget] = 0
    done = False
    base_command = [os.path.abspath(env['mpy_path']), os.path.abspath(env['mpy_main']), '-m', 'random', '-o', 'screenshot.jpg', '-W', str(args.width), '-H', str(args.height), '-c', str(args.count), '-l', args.layout, '--normalize']
    iteration = 0
    errors = 0
    files = []
    while not done:
        iteration += 1
        command = base_command + ['-t', *[widget for widget in widgets.keys() if widgets[widget] <= args.amount]]
        print(f"Running command: {' '.join(command)}")
        gen = subprocess.run(args=command, cwd=os.path.abspath(os.path.curdir), capture_output=True, text=True)
        if gen.returncode != 0:
            print(f"Failed to generate random UI in iteration {iteration}:\n{gen.stdout}\n{gen.stderr}")
            errors += 1
            continue
        tmp_image = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), "screenshot.jpg"))
        tmp_text = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), "screenshot.txt"))
        if not os.path.exists(tmp_image) or not os.path.exists(tmp_text):
            print(f"Failed to find generated random UI files in iteration {iteration}")
            errors += 1
            continue
        gen_image = os.path.abspath(os.path.join(env['output_folder'], f"ui_{iteration}.jpg"))
        gen_text = os.path.abspath(os.path.join(env['output_folder'], f"ui_{iteration}.txt"))
        try:
            shutil.move(tmp_image, gen_image)
            shutil.move(tmp_text, gen_text)
        except FileNotFoundError as e:
            print(f"Failed to move files in iteration {iteration}:\n{tmp_image} -> {gen_image}\n{tmp_text} -> {gen_text}\n{e}")
            errors += 1
            continue
        files.append((gen_image, gen_text))
        with open(gen_text, 'r') as f:
            # Each line is in this format: "class x y w h" (Need to grab class)
            for line in f.readlines():
                widget = line.split(' ')[0]
                if widget in widgets:
                    widgets[widget] += 1
                else:
                    errors += 1
                    print(f"Unknown widget class {widget} found in annotation file of iteration {iteration}")
        logger.report_scalar(title='Generator', series='total_widgets', value=sum(widgets.values()), iteration=iteration)
        logger.report_scalar(title='Generator', series='errors', value=errors, iteration=iteration)
        for widget in widgets:
            logger.report_scalar(title='Widget metrics', series=widget, value=widgets[widget], iteration=iteration)
        if all([widgets[widget] >= args.amount for widget in widgets]):
            done = True
    generated_widgets = args.amount * len(implemented_types)
    generated_files = len(files)
    logger.report_histogram(title='Generated widgets', series='total', values=np.array([[generated_widgets], [generated_files]]), labels=['Widgets', 'Files'], yaxis='Count')
    print("Finished generating random UIs")
    env['generated_widgets'] = generated_widgets
    env['generated_files'] = generated_files
    env['files'] = files


def capture_design(design_folder: str):
    import subprocess, os, shutil
    from clearml import Task
    task = Task.current_task()

# Dataset helpers
def replace_class_names(files: list):
    global implemented_types
    for i,(_, annotation) in enumerate(files):
        for a_class in implemented_types:
            replacement = str(implemented_types.index(a_class))
            with open(annotation, 'r+') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    line = line.replace(a_class + " ", replacement + " ")
                    lines[i] = line
                f.seek(0)
                f.writelines(lines)
                f.truncate()

def shuffle_split(input: list, split_ratio: tuple):
    import random
    if len(split_ratio) != 3 or sum(split_ratio) != 1:
        raise ValueError("Split ratio must be a tuple of 3 values that sum up to 1. (e.g. (0.7, 0.1, 0.2))")
    random.shuffle(input)
    part1 = int(len(input) * split_ratio[0])
    part2 = int(len(input) * split_ratio[1])
    # NOTE Part3 is the remainder of the input

    split1 = input[:part1]
    split2 = input[part1:part1 + part2]
    split3 = input[part1 + part2:]
    return (split1, split2, split3)

def prepare_dataset(env: dict, args: argparse.Namespace):
    import os, yaml, shutil
    from clearml import Task
    import numpy as np
    task = Task.current_task()
    global implemented_types
    train_img_dir = "images/train"
    train_label_dir = "labels/train"
    val_img_dir = "images/val"
    val_label_dir = "labels/val"
    test_img_dir = "images/test"
    test_label_dir = "labels/test"
    target_dir = os.path.join(env['output_folder'], args.type)
    dataset_file = os.path.abspath(os.path.join(env['output_folder'], args.type, f"{args.type}.yaml"))
    train_img_folder = os.path.join(target_dir, train_img_dir)
    train_label_folder = os.path.join(target_dir, train_label_dir)
    val_img_folder = os.path.join(target_dir, val_img_dir)
    val_label_folder = os.path.join(target_dir, val_label_dir)
    test_img_folder = os.path.join(target_dir, test_img_dir)
    test_label_folder = os.path.join(target_dir, test_label_dir)
    folders = [target_dir, train_img_folder, train_label_folder, val_img_folder, val_label_folder, test_img_folder, test_label_folder]
    # Prepare files
    replace_class_names(env['files'])
    split_ratio = args.split_ratio if args.split_ratio else (0.7, 0.1, 0.2)
    train_files, val_files, test_files = shuffle_split(env['files'], split_ratio)
    # Prepare folders
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    # Move files
    for i, (image, annotation) in enumerate(train_files):
        shutil.move(image, os.path.join(train_img_folder, f"train_{i}.jpg"))
        shutil.move(annotation, os.path.join(train_label_folder, f"train_{i}.txt"))
    for i, (image, annotation) in enumerate(val_files):
        shutil.move(image, os.path.join(val_img_folder, f"val_{i}.jpg"))
        shutil.move(annotation, os.path.join(val_label_folder, f"val_{i}.txt"))
    for i, (image, annotation) in enumerate(test_files):
        shutil.move(image, os.path.join(test_img_folder, f"test_{i}.jpg"))
        shutil.move(annotation, os.path.join(test_label_folder, f"test_{i}.txt"))
    # Create dataset YAML
    content = {
        'path': os.path.join(env['output_folder'], args.type),
        'train': train_img_dir,
        'val': val_img_dir,
        'test': test_img_dir,
        'names': implemented_types
    }
    with open(dataset_file, 'w') as f:
        yaml.dump(content, f)
    task.get_logger().report_histogram(title='Split (train, val, test)', series='total', values=np.array([[len(train_files)], [len(val_files)], [len(test_files)]]), labels=['Train', 'Validate', 'Test'], yaxis='File count')
    env['dataset_folder'] = target_dir
    env['metadata'] = {'Generator type': args.type, 'Split ratio': split_ratio, 'Total files': len(env['files']), 'Train': len(train_files), 'Validate': len(val_files), 'Test': len(test_files)}
    # Alternative artifact upload to task
    if args.no_dataset:
        task.upload_artifact('Dataset', target_dir, metadata=env['metadata'])

def create_dataset(env: dict, args: argparse.Namespace):
    from clearml import Task, Dataset
    task = Task.current_task()
    if not args.dataset:
        args.dataset = f"LVGL UI (instances={env['generated_widgets']},files={env['generated_files']})"
    else:
        args.dataset = f"{args.dataset} (instances={env['generated_widgets']},files={env['generated_files']})"
    task.rename(f"{args.type.upper()} UI Dataset")
    dataset = Dataset.create(dataset_name=args.dataset, dataset_project="LVGL UI Detector", dataset_tags=["lvgl-ui-detector", args.type], use_current_task=True)
    dataset.add_files(env['dataset_folder'])
    dataset.upload()
    comment = f"Dataset '{args.dataset}' created: {dataset.id}"
    print(comment)
    dataset.set_metadata(env['metadata'])
    dataset.finalize()
    task.set_comment(comment)

# Environment helpers
def prepare_task(args: dict):
    from clearml import Task
    task = Task.init(project_name='LVGL UI Detector', task_name=f'UI Generator', task_type=Task.TaskTypes.data_processing)
    task.connect(args)

def prepare_environment(output_folder: str, mpy_path: str, mpy_main: str):
    import os
    env = {
        'output_folder': os.path.abspath(output_folder),
        'mpy_path': os.path.abspath(mpy_path),
        'mpy_main': os.path.abspath(mpy_main)
    }
    os.makedirs(env['output_folder'], exist_ok=True)
    return env

# Design Mode helpers (for ChatGPT generation)
def load_json_file(filepath: str):
    import json
    with open(filepath, 'r') as f:
        return json.load(f)
def verify_design(design_file: str, schema_file: str) -> bool|tuple[bool, Exception]:
    from jsonschema import validate
    from jsonschema.exceptions import ValidationError
    design = load_json_file(design_file)
    schema = load_json_file(schema_file)
    try:
        validate(instance=design, schema=schema)
        print(f"Provided design file {design_file} is valid.")
        return True
    except ValidationError as e:
        print(f"Provided design file {design_file} is invalid:\n{e}")
        return False, e

def generate_designs(env: dict, args: argparse.Namespace):
    import os
    from openai import OpenAI
    from openai.types import ChatModel
    open_ai = {}
    if args.model not in ChatModel:
        print(f"Invalid model name: {args.model}. Valid are:\n{'\n'.join(ChatModel)}")
        return
    client = OpenAI() # per-default uses api_key=os.environ['OPENAI_API_KEY']
    env['OPEN_AI'] = open_ai

# CLI helpers
class _HelpAction(argparse._HelpAction):
    # Source: https://stackoverflow.com/a/24122778
    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()

        # retrieve subparsers from parser
        subparsers_actions = [
            action for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)]
        # there will probably only be one subparser_action,
        # but better save than sorry
        for subparsers_action in subparsers_actions:
            # get all subparsers and print help
            for choice, subparser in subparsers_action.choices.items():
                print("Subparser '{}'".format(choice))
                print(subparser.format_help())

        parser.exit()

def cli_parser():
    parser = argparse.ArgumentParser(description='Generate UI Detector dataset', add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-h', '--help', action=_HelpAction, help='show this help message and exit') # add custom help
    parser.add_argument('-o', '--output', type=str, default='output', help='Output folder')
    parser.add_argument('--mpy-path', type=str, default='', help='Path to MicroPython binary')
    parser.add_argument('--mpy-main', type=str, default='', help='Path to main.py of micropython script')
    parser.add_argument('-d', '--dataset', type=str, default='', help='Name of the dataset')
    parser.add_argument('-s', '--split-ratio', type=int, nargs=3, default=None, help='Split ratio for train, val, test')
    parser.add_argument('--no-dataset', action='store_true', help='Do not create a ClearML dataset (artifacts are added to Task)')
    type = parser.add_subparsers(dest='type', help='Type of LVGL UI generator to use')
    random_gen = type.add_parser('random', help='Generate random UI')
    random_gen.add_argument('-W', '--width', type=int, default=250, help='Width of the UI screenshot')
    random_gen.add_argument('-H', '--height', type=int, default=250, help='Height of the UI screenshot')
    random_gen.add_argument('-c', '--count', type=int, default=3, help='Number of widgets to create per output')
    random_gen.add_argument('-l', '--layout', type=str, default='none', help='The main container layout of the random UI ["grid", "flex", "none"]')
    random_gen.add_argument('-x', '--amount', type=int, default=50, help='Amount of outputs per widget class to create')
    design_gen = type.add_parser('design', help='Generate UI from design files')
    design_variant = design_gen.add_subparsers(dest='variant', help='Variant of design generator to use')
    design_local = design_variant.add_parser('local', help='Generate UIs from local design files')
    design_local.add_argument('-f', '--design-folder', type=str, help='Source folder for design generator')
    design_remote = design_variant.add_parser('remote', help='Generate UIs from remote design files')
    # TODO Add arguments for remote design generator
    design_gpt = design_variant.add_parser('gpt', help='Generate UIs using ChatGPT API')
    design_gpt.add_argument('--api-key', type=str, required=True, help='ChatGPT API key')
    design_gpt.add_argument('--model', type=str, required=True, help='ChatGPT model name')
    design_gpt.add_argument('--max-tokens', type=int, default=255, help=f'ChatGPT maximum tokens') # TODO Check what would be sensible amount of default tokens
    design_gpt.add_argument('--designs', type=int, default=10, help='Number of designs to generate')
    design_gpt.add_argument('-o', '--output-folder', type=str, default='tmp', help='Output folder for generated designs')
    gpt_arg = design_gpt.add_mutually_exclusive_group()
    gpt_arg.add_argument('--temperature', type=float, default=0.7, help='ChatGPT sampling temperature')
    gpt_arg.add_argument('--top-p', type=float, default=1.0, help='ChatGPT top-p sampling')
    return parser

def validate_cli_args(args: dict):
    import os, sys
    has_error = False # NOTE Uses flag to check for multiple argument errors at once and then exit (more useful when running as task)
    if not args.mpy_path:
        if 'MICROPYTHON_BIN' in os.environ:
            args.mpy_path = os.environ['MICROPYTHON_BIN']
        else:
            print('MicroPython path not provided and MICROPYTHON_BIN environment variable not set')
            has_error = True
    if not args.mpy_main:
        if 'MICROPYTHON_MAIN' in os.environ:
            args.mpy_main = os.environ['MICROPYTHON_MAIN']
        else:
            print('MicroPython main.py path not provided and MICROPYTHON_MAIN environment variable not set')
            has_error = True
    if not os.path.exists(args.mpy_path):
        print(f'MicroPython path {args.mpy_path} does not exist')
        has_error = True
    if not os.path.exists(args.mpy_main):
        print(f'MicroPython main.py path {args.mpy_main} does not exist')
        has_error = True
    if args.type == 'design' and args.variant == 'local':
        if not args.design_folder:
            print('Design folder not provided for local design generator')
            has_error = True
    if args.type == 'design' and args.variant == 'remote':
        # TODO Add validation for remote design generator
        raise NotImplementedError("Remote design generator not implemented yet")
        pass
    if args.type == 'design' and args.variant == 'gpt':
        if not args.api_key:
            print('ChatGPT API key not provided')
            has_error = True
        if not args.model:
            print('ChatGPT model name not provided')
            has_error = True
    if has_error:
        sys.exit(1)

def verify_cli_args(args: dict):
    import os, sys

if __name__ == '__main__':
    parser = cli_parser()
    args = parser.parse_args()
    prepare_task(args)
    validate_cli_args(args)
    env = prepare_environment(args.output, args.mpy_path, args.mpy_main)
    if args.type == 'random':
        capture_random(env, args)
    elif args.type == 'design':
        if args.variant == 'local':
            capture_design(env, args)
        elif args.variant == 'remote':
            pass
        elif args.variant == 'gpt':
            generate_designs(env, args)
            capture_design(env, args)
    prepare_dataset(env, args)
    if not args.no_dataset:
        create_dataset(env, args)