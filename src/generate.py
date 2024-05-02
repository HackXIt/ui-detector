import argparse
from importlib.metadata import metadata
from os import system
from pyexpat.errors import messages
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
    import subprocess, os, shutil
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
    logger = task.get_logger()
    design_files = [f for f in os.listdir(design_folder) if os.path.isfile(os.path.join(design_folder, f))]
    files = []
    errors = 0
    for i, design_file in enumerate(design_files):
        print(f"Running design generator on file {design_file}")
        gen = subprocess.run([os.path.abspath(env['mpy_path']), os.path.abspath(env['mpy_main']), '-m', 'design', '-o', 'screenshot.jpg', '-f', os.path.abspath(os.path.join(design_folder, design_file)), '--normalize'], cwd=os.path.abspath(os.path.curdir), capture_output=True, text=True)
        if gen.returncode != 0:
            print(f"Failed to generate UI from design file {design_file}:\n{gen.stdout}\n{gen.stderr}")
            errors += 1
            continue
        tmp_image = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), "screenshot.jpg"))
        tmp_text = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), "screenshot.txt"))
        if not os.path.exists(tmp_image) or not os.path.exists(tmp_text):
            print(f"Failed to find generated UI files from design file {design_file}")
            errors += 1
            continue
        gen_image = os.path.abspath(os.path.join(env['output_folder'], f"ui_{i}.jpg"))
        gen_text = os.path.abspath(os.path.join(env['output_folder'], f"ui_{i}.txt"))
        try:
            shutil.move(tmp_image, gen_image)
            shutil.move(tmp_text, gen_text)
        except FileNotFoundError as e:
            print(f"Failed to move files from design file {design_file}:\n{tmp_image} -> {gen_image}\n{tmp_text} -> {gen_text}\n{e}")
            errors += 1
            continue
        files.append((gen_image, gen_text))
        logger.report_scalar(title='Generator', series='errors', value=errors, iteration=i)
    generated_files = len(files)
    env['generated_files'] = generated_files
    env['files'] = files

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

def verify_design_from_file(design_file: str, schema_file: str) -> tuple[bool, Exception]:
    from jsonschema import validate
    from jsonschema.exceptions import ValidationError
    design = load_json_file(design_file)
    schema = load_json_file(schema_file)
    try:
        validate(instance=design, schema=schema)
        print(f"Provided design file {design_file} is valid.")
        return True, None
    except ValidationError as e:
        print(f"Provided design file {design_file} is invalid:\n{e}")
        return False, e

def verify_design_from_string(design: str, schema_file: str) -> tuple[bool, Exception]:
    from jsonschema import validate
    from jsonschema.exceptions import ValidationError
    schema = load_json_file(schema_file)
    try:
        validate(instance=design, schema=schema)
        return True, None
    except ValidationError as e:
        return False, e


def ask_gpt(openai: dict):
    from openai import OpenAI
    client = OpenAI() # per-default uses api_key=os.environ['OPENAI_API_KEY']
    if 'temperature' in openai:
        response = client.chat.completions.create(
            model=openai['model'],
            messages=openai['messages'],
            max_tokens=openai['max_tokens'],
            response_format={"type": openai['format']},
            stop=openai['stop'] if 'stop' in openai else None,
            temperature=openai['temperature'],
        )
    elif 'top_p' in openai:
        response = client.chat.completions.create(
            model=openai['model'],
            messages=openai['messages'],
            max_tokens=openai['max_tokens'],
            response_format={"type": openai['format']},
            stop=openai['stop'] if 'stop' in openai else None,
            top_p=openai['top_p']
        )
    else:
        response = client.chat.completions.create(
            model=openai['model'],
            messages=openai['messages'],
            max_tokens=openai['max_tokens'],
            response_format={"type": openai['format']},
            stop=openai['stop'] if 'stop' in openai else None,
        )
    # TODO Return the response conditionally based on the choices made or none on failure
    return response

def ask_gpt_for_design_idea(openai: dict):
    system_message = """
You are a UI designer.
Your goal is to create a new design idea and guideline for an user interface to be given to developers.
The user will provide a context constraint and goal for the design.
You must adhere to the design constraints and goals.
Make sure to include the following in your design idea:
- A title for the design
- The context of the design
- The design goals or objectives
- The design elements
- A high-level description of the design

DESIGN RULES:
In the idea, you will include mentions of design elements that should be present in the form of widgets.
These are the allowed widget types to be used as design elements and their corresponding purposes:
- Arc: A circular arc to select or display a value within a range
- Bar: A horizontal or vertical bar to select or display a value within a range
- Button: Clickable button with or without text
- ButtonMatrix: Matrix of buttons with or without text
- Calendar: A calendar showing the month with or without highlighted dates
- Checkbox: A box that can be checked or unchecked
- Dropdown: A box containing multiple options as entries where one can be selected
- Label: A field for text display
- LED: A circle with custom styling that can be turned on or off like a light
- Roller: A special dropdown with a rolling interface for selecting entries
- Scale: A scale for displaying a range of values
- Slider: A slider for selecting a value within a range
- Spinbox: Spinbox widget
- Switch: A switch for toggling between two states
- Table: A tabular widget for displaying data
- TextArea: A field for input or display of multiple text lines
- Container: A special widget containing other widgets in a structured layout

Design elements can be structurally combined using the special widget 'container'.
The structure of such a container may be defined as a layout of the following types and purposes:
- none: Widgets are structured using absolute positioning
- grid: Widgets are structured in rows and columns, each cell containing a widget
- flex: Widgets are structured in a flexible layout that adjusts to the available space horizontally or vertically

Design elements can also be styled individually or identical using a style group.
Style groups define a set of style properties that can be applied to multiple widgets at once.
The style properties include color, size, padding, margin, opacity, border, lines, shadows and more.
Such style groups are named using a unique identifier and may be applied to multiple widgets, but each widget only can have one style group applied.

TASK:
It is your task to create a design idea for a user interface that adheres to these design rules.
"""
    initial_prompt = """
Create a new UI design in the context of an embedded environment with a single display.
"""
    openai['messages'] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": initial_prompt}
    ]
    openai['format'] = "text"
    # Get response
    response = ask_gpt(openai)
    return response.choices[0].message.content

def ask_gpt_for_design_json(openai: dict, design: dict):
    import os, json
    schema_file = os.path.join(os.path.curdir, 'schema', 'design_file.schema.json')
    openai['schema'] = json.load(open(schema_file, 'r'))
    system_message = """
You are a UI generator.
Your goal is to create a new single window UI using a specialized JSON format.
The format specification is available in the design.schema.json file.
Follow the design guideline of the user when generating the UI.
Output a valid JSON object that represents the UI design.
    """
    system_message += f"\ndesign.schema.json:\n{str(openai['schema'])}"
    initial_prompt = "Create a new UI design from this design idea:\n"
    if len(design['errors']) == 0:
        openai['messages'] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": initial_prompt + design['idea']},
        ]
    else:
        # NOTE This is a bit of a hack using only the last error message (which might result in error repetition)
        error_prompt = "The design has validation errors:\n" + str(design['errors'][-1]) + "\nCorrect ALL errors and generate a complete JSON output."
        openai['messages'] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": initial_prompt + design['idea']},
            {"role": "assistant", "content": design['json_raw']},
            {"role": "user", "content": error_prompt}
        ]
    openai['format'] = "json_object"
    # Get response
    response = ask_gpt(openai)
    json_design = response.choices[0].message.content
    json_file = os.path.join(os.path.curdir, 'tmp', f"design_{design['attempts']}.json")
    with open(json_file, 'w') as f:
        f.write(json_design)
    # Check if response is valid JSON
    is_valid, error = verify_design_from_file(json_file, schema_file) # NOTE Ignoring the error, could be useful for debugging though
    return is_valid, json_design, error

def generate_designs(env: dict, args: argparse.Namespace):
    import os, json
    # TODO should connect stuff to ClearML task
    open_ai = {}
    os.makedirs(args.output_folder, exist_ok=True)
    env['design_folder'] = args.output_folder
    open_ai['model'] = args.model
    open_ai['max_tokens'] = args.max_tokens
    if hasattr(args, 'temperature'):
        open_ai['temperature'] = args.temperature
    elif hasattr(args, 'top_p'):
        open_ai['top_p'] = args.top_p
    for i in range(args.designs):
        print(f"Generating design {i}...")
        valid_design = False
        design = {}
        design['attempts'] = 0
        design['errors'] = []
        design['idea'] = ask_gpt_for_design_idea(open_ai)
        print(f"Idea: {design['idea']}")
        while not valid_design and design['attempts'] < 3: # NOTE Limiting attempts to 5 (limiting costs for now)
            valid_design, design['json_raw'], error = ask_gpt_for_design_json(open_ai, design)
            if error:
                design['errors'].append(error)
                design['attempts'] += 1
                print(f"JSON generation failed with error.")
                with open(os.path.join(os.path.curdir, 'tmp', f"design_{i}_error.txt"), 'w+') as f:
                    f.write(str(error))
        else:
            msg_fail = f"Failed to generate valid design after {design['attempts']} attempts"
            msg_success = f"Generated valid design after {design['attempts']}"
            print(msg_fail if not valid_design else msg_success)
        print(f"Design JSON:\n{design['json_raw']}")
        design['file'] = os.path.join(args.output_folder, f"design_{i}.json")
        with open(design['file'], 'w') as f:
            json.dump(design['json_raw'], f)
        open_ai[f'design_{i}'] = design
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
    parser.add_argument('-o', '--output', type=str, default='tmp/output', help='Output folder')
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
    design_gpt.add_argument('--api-key', type=str, default='', help='ChatGPT API key')
    design_gpt.add_argument('--model', type=str, required=True, help='ChatGPT model name')
    design_gpt.add_argument('--max-tokens', type=int, default=2500, help=f'ChatGPT maximum tokens') # TODO Check what would be sensible amount of default tokens
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
        if not args.api_key and 'OPENAI_API_KEY' not in os.environ:
            print('ChatGPT API key not provided')
            has_error = True
        if not args.model:
            print('ChatGPT model name not provided')
            has_error = True
    if has_error:
        sys.exit(1)

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
            capture_design(env['output_folder'])
        elif args.variant == 'remote':
            pass
        elif args.variant == 'gpt':
            generate_designs(env, args)
            capture_design(env['output_folder'])
    prepare_dataset(env, args)
    if not args.no_dataset:
        create_dataset(env, args)