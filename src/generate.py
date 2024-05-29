"""
This script contains the generation process pipelines for the LVGL UI Generator project.
It depends on version 2 of the generator to work.
It will call the Micropython generator script to create UIs based on the provided arguments.
The script will generate a dataset of UIs based on the chosen mode (random or design) and their corresponding arguments.

The script may be called according to the following usage information:

    usage: generate.py [-h] [-o OUTPUT_FOLDER] [--mpy-path MPY_PATH] [--mpy-main MPY_MAIN] [-d DATASET] [-s SPLIT_RATIO SPLIT_RATIO SPLIT_RATIO] [--no-dataset]
                    {random,design} ...

    Generate UI Detector dataset

    positional arguments:
    {random,design}       Type of LVGL UI generator to use
        random              Generate random UI
        design              Generate UI from design files

    options:
    -h, --help            show this help message and exit
    -o OUTPUT_FOLDER, --output-folder OUTPUT_FOLDER
                            Output folder (default: tmp/output)
    --mpy-path MPY_PATH   Path to MicroPython binary (loads from environment MICROPYTHON_BIN if not provided) (default: )
    --mpy-main MPY_MAIN   Path to main.py of micropython script (loads from environment MICROPYTHON_MAIN if not provided) (default: )
    -d DATASET, --dataset DATASET
                            Custom name of the dataset written in the task comment (default: )
    -s SPLIT_RATIO SPLIT_RATIO SPLIT_RATIO, --split-ratio SPLIT_RATIO SPLIT_RATIO SPLIT_RATIO
                            Split ratio for train, val, test (default: None)
    --no-dataset          Do not create a ClearML dataset (artifacts are added to Task) (default: False)
    Subparser 'random'
    usage: generate.py random [-h] [-W WIDTH] [-H HEIGHT] [-c COUNT] [-l LAYOUT] [-x AMOUNT]

    options:
    -h, --help            show this help message and exit
    -W WIDTH, --width WIDTH
                            Width of the UI screenshot
    -H HEIGHT, --height HEIGHT
                            Height of the UI screenshot
    -c COUNT, --count COUNT
                            Number of widgets to create per output
    -l LAYOUT, --layout LAYOUT
                            The main container layout of the random UI ["grid", "flex", "none"]
    -x AMOUNT, --amount AMOUNT
                            Amount of outputs per widget class to create

    Subparser 'design'
    usage: generate.py design [-h] {local,remote,gpt} ...

    positional arguments:
    {local,remote,gpt}  Variant of design generator to use
        local             Generate UIs from local design files
        remote            Generate UIs from remote design files
        gpt               Generate UIs using ChatGPT API

    options:
    -h, --help          show this help message and exit
"""

import argparse

topics = [
    "Smart Home Control Systems",
    "Wearable Fitness Trackers",
    "Automotive Dashboard Displays",
    "Industrial Automation Interfaces",
    "Agricultural Monitoring Systems",
    "Medical Device Interfaces",
    "Drone Control Panels",
    "Retail Point of Sale Systems",
    "Smart Watches Interfaces",
    "Security System Controls",
    "Marine Navigation Systems",
    "Building Climate Control Systems",
    "Home Appliance Controls (e.g., Smart Refrigerators)",
    "Energy Management Displays",
    "Portable Music Players",
    "Electronic Thermostats",
    "Educational Tablets for Kids",
    "Emergency Alert Systems",
    "Water Purification System Controls",
    "Lighting Control Systems",
    "Portable Gaming Devices",
    "Smart Mirror Technologies",
    "Elevator Control Panels",
    "Vending Machine Interfaces",
    "Fitness Equipment Consoles",
    "Industrial Robot Controllers",
    "Smart Bed Controls",
    "Smart Glasses Interfaces",
    "Pet Tracking Devices",
    "Baby Monitoring Systems",
    "Digital Signage",
    "Ticketing Kiosks",
    "Virtual Reality Headset Interfaces",
    "Library Management Kiosks",
    "Smart Lock Interfaces",
    "Laboratory Equipment Interfaces",
    "Smart Pens",
    "Art Installation Controls",
    "HVAC Control Systems",
    "Railroad Monitoring Systems",
    "Handheld GPS Devices",
    "Digital Cameras",
    "Smart Toothbrushes",
    "Aircraft Cockpit Displays",
    "Electric Vehicle Charging Stations",
    "Soil Moisture Sensors",
    "Smart Jewelry",
    "Pipeline Monitoring Systems",
    "Waste Management Systems",
    "Personal Medical Devices (e.g., Insulin Pumps)",
    "Public Transportation Displays",
    "On-board Ship Computers",
    "Smart Plant Pots",
    "Industrial Pressure Sensors",
    "Interactive Museum Exhibits",
    "Smart Bicycle Systems",
    "Conference Room Booking Displays",
    "Augmented Reality Interfaces",
    "Remote Wilderness Cameras",
    "Interactive Retail Displays",
    "Spacecraft Control Interfaces",
    "Wireless Router Management",
    "Smart City Infrastructure Interfaces",
    "Factory Assembly Line Displays",
    "Car Rental Kiosks",
    "Airport Check-in Kiosks",
    "Digital Billboards",
    "Hospital Room Information Panels",
    "Power Grid Monitoring Systems",
    "Oil Rig Monitoring Interfaces",
    "Smart Suitcases",
    "Fishing Gear Electronics",
    "Underwater Exploration Devices",
    "Digital Menu Boards in Restaurants",
    "Emergency Vehicle Dashboards",
    "Voice-Controlled Home Assistants",
    "Smart Coasters (beverage temperature)",
    "Bicycle Sharing System Terminals",
    "Smart Shower Panels",
    "Mining Equipment Interfaces",
    "Forest Fire Detection Systems",
    "Smart Windows",
    "Interactive Dance Floors",
    "Smart Ring Interfaces",
    "Professional Camera Systems",
    "Home Brewing Systems",
    "Smart Mailboxes",
    "Autonomous Farm Equipment",
    "Wind Turbine Controls",
    "Smart Blinds and Curtains",
    "Logistics Tracking Systems",
    "Parking Garage Equipment",
    "Smart Helmet Displays",
    "Boat Instrumentation Panels",
    "Interactive Park Equipment",
    "Livestock Tracking Systems",
    "Remote Surgery Consoles",
    "Weather Monitoring Stations",
    "Smart Gloves",
    "Electronic Voting Machines"
]
"""
100 topics that an embedded user interface could be about.
It was generated using the GPT model from OpenAI with the prompt "Create a list of 100 topics that an embedded user interface could be about."
"""

themes = [
    " Minimalist",
    "Futuristic",
    "Retro",
    "High Contrast",
    "Dark Mode",
    "Light Mode",
    "Nature-inspired",
    "Nautical",
    "Neon Glow",
    "Earthy Tones",
    "Pastel Colors",
    "High Tech",
    "Art Deco",
    "Steampunk",
    "Material Design",
    "Flat Design",
    "3D Depth",
    "Monochrome",
    "Kids-Friendly",
    "Elderly-Friendly",
    "Luxury",
    "Industrial",
    "Sports",
    "Educational",
    "Seasonal (e.g., Winter, Summer)",
    "Holiday Themes (e.g., Christmas, Halloween)",
    "Cartoon",
    "Abstract",
    "Photorealistic",
    "Geometric",
    "Military",
    "Space Exploration",
    "Underwater",
    "Urban",
    "Rural",
    "Health Focused",
    "Accessibility Enhanced",
    "Cultural (e.g., Japanese, Mexican)",
    "Cyberpunk",
    "Virtual Reality",
    "Augmented Reality",
    "Transparent Interfaces",
    "Glass Effect",
    "Vintage Film",
    "Comic Book",
    "Parchment and Ink",
    "Origami",
    "Glow in the Dark",
    "Neon Signs",
    "Hand-drawn",
    "Watercolor",
    "Grunge",
    "Metallic",
    "Zen and Tranquility",
    "Casino",
    "Outer Space",
    "Sci-Fi",
    "Historical Periods (e.g., Victorian, Medieval)",
    "Typography-Based",
    "Animal Print",
    "Floral",
    "Ocean Waves",
    "Desert Sands",
    "Mountainous Terrain",
    "Tropical Paradise",
    "Arctic Freeze",
    "Jungle Theme",
    "Auto Racing",
    "Aviation",
    "Sailing",
    "Rock and Roll",
    "Hip Hop",
    "Classical Music",
    "Opera",
    "Ballet",
    "Theatre",
    "Film Noir",
    "Silent Film",
    "Neon Jungle",
    "Crystal Clear",
    "Witchcraft and Wizardry",
    "Steampunk Mechanisms",
    "Pop Art",
    "Renaissance Art",
    "Graffiti",
    "Pixel Art",
    "ASCII Art",
    "Mosaic",
    "Lego Style",
    "Board Game",
    "Video Game",
    "Dystopian",
    "Utopian",
    "Western",
    "Eastern",
    "Minimalist Text",
    "Bold Color Blocks",
    "Line Art",
    "Optical Illusions",
    "Neon Abstract"
]
"""
100 themes that could be applied to user interfaces.
It was generated using the GPT model from OpenAI with the prompt "Create a list of 100 themes that could be applied to user interfaces."
"""

combinations = [(t, c) for t in themes for c in topics]
"""
All possible combinations of themes and topics for the LVGL UI Generator.
With 100 themes and 100 topics, there are 10,000 possible combinations.
"""

implemented_types = ["arc", 
                     "bar", "button", "buttonmatrix", 
                     "calendar", "checkbox", 
                     "dropdown", 
                     "label",
                     "roller", 
                     "scale", "slider", "spinbox", "switch", 
                     "table", "textarea"]
"""
The implemented and used widget types for the LVGL UI Generator.
"""

# Generators
def capture_random(env: dict, args: argparse.Namespace):
    """
    **Params:**
    - `env` Environment dictionary containing the output folder and Micropython paths
    - `args` Arguments containing the width, height, count, layout, and amount of widgets to generate

    Calls the Micropython generator using random mode to create a user interface.

    In each iteration, the generator is provided with a list of widgets, which it will use to create a randomized UI.
    Each generated UI is saved as an JPEG image, with a text file containing the label annotations.

    Labels are in the format: "class x y w h" (class is the widget type, x/y are the center coordinates, w/h are the width/height)
    Label values are normalized to the range [0.0, 1.0].
    After each generation, the contained widgets are counted and their total count is accumulated.
    Generation is repeated until the desired threshhold per widget is reached.
    Labels will be post-processed immediatly, removing any invalid annotations (out-of-bounds of the window).
    The generation process is tracked and reported to the ClearML task.

    A histogram is created to show the total amount of widgets generated and the amount of files created.
    """
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
        annotation_errors = []
        with open(gen_text, 'r+') as f:
            # Each line is in this format: "class x y w h" (Need to grab class)
            new_lines = []
            for i, line in enumerate(f.readlines()):
                widget, x, y, w, h = line.split(' ')
                x, y, w, h = float(x), float(y), float(w), float(h)
                if any([x < 0.0, y < 0.0, w < 0.0, h < 0.0]) or any([x > 1.0, y > 1.0, w > 1.0, h > 1.0]):
                    errors += 1
                    print(f"[Line {i}] Invalid bounding box found in annotation file {gen_text} of iteration {iteration}")
                    print(f"Removed: {widget} {x} {y} {w} {h}")
                    annotation_errors.append(i)
                    continue
                new_lines.append(line)
                if widget in widgets:
                    widgets[widget] += 1
                else:
                    errors += 1
                    print(f"[Line {i}] Unknown widget class {widget} found in annotation file of iteration {iteration}")
            # NOTE Delete invalid annotations in label file
            f.seek(0)
            f.writelines(new_lines)
            f.truncate()
            del new_lines
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
    """
    **Params:**
    - `design_folder` Folder containing the design files to generate UIs from

    Calls the Micropython generator using design mode to create a user interface.

    In each iteration, the generator is called with a design file from the provided folder.
    Each generated UI is saved as an JPEG image, with a text file containing the label annotations.

    Labels are in the format: "class x y w h" (class is the widget type, x/y are the center coordinates, w/h are the width/height)
    Label values are normalized to the range [0.0, 1.0].
    After each generation, the contained widgets are counted and their total count is accumulated.
    Generation is repeated until the desired threshhold per widget is reached.
    Labels will be post-processed immediatly, removing any invalid annotations (out-of-bounds of the window).
    The generation process is tracked and reported to the ClearML task.

    A histogram is created to show the total amount of widgets generated and the amount of files created.
    """
    import subprocess, os, shutil
    from clearml import Task
    print(f"Capturing designs from {design_folder}...")
    task = Task.current_task()
    logger = task.get_logger()
    design_files = [f for f in os.listdir(design_folder) if os.path.isfile(os.path.join(design_folder, f))]
    if len(design_files) == 0:
        print(f"No design files found in {design_folder}")
        return
    widgets = {}
    for widget in implemented_types:
        widgets[widget] = 0
    files = []
    errors = 0
    logger.report_scalar(title='Generator', series='total_widgets', value=sum(widgets.values()), iteration=0)
    logger.report_scalar(title='Generator', series='errors', value=errors, iteration=0)
    for widget in widgets:
        logger.report_scalar(title='Widget metrics', series=widget, value=widgets[widget], iteration=0)
    for i, design_file in enumerate(design_files):
        print(f"Iteration: {i+1}/{len(design_files)} - {design_file}")
        attempts = 0
        success = False
        # NOTE Retry mechanism due to possible MemoryErrors when dynamically allocating screenshot data (Trust in the OS to clean up the mess)
        while not success and attempts < 4:
            print(f"Running design generator on file {design_file}")
            gen = subprocess.run([os.path.abspath(env['mpy_path']), os.path.abspath(env['mpy_main']), '-m', 'design', '-o', 'screenshot.jpg', '-f', os.path.abspath(os.path.join(design_folder, design_file)), '--normalize'], cwd=os.path.abspath(os.path.curdir), capture_output=True, text=True)
            if gen.returncode != 0:
                print(f"Failed to generate UI from design file {design_file}:\n{gen.stdout}\n{gen.stderr}")
                attempts += 1
                continue
            success = True
        if not success:
            print(f"Failed to generate UI from design file {design_file} after {attempts} attempts")
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
            print(f"{design_file} -> {gen_image}")
            shutil.move(tmp_image, gen_image)
            shutil.move(tmp_text, gen_text)
        except FileNotFoundError as e:
            print(f"Failed to move files from design file {design_file}:\n{tmp_image} -> {gen_image}\n{tmp_text} -> {gen_text}\n{e}")
            errors += 1
            continue
        files.append((gen_image, gen_text))
        annotation_errors = []
        with open(gen_text, 'r+') as f:
            # Each line is in this format: "class x y w h" (Need to grab class)
            new_lines = []
            for i, line in enumerate(f.readlines()):
                widget, x, y, w, h = line.split(' ')
                x, y, w, h = float(x), float(y), float(w), float(h)
                if any([x < 0.0, y < 0.0, w < 0.0, h < 0.0]) or any([x > 1.0, y > 1.0, w > 1.0, h > 1.0]):
                    errors += 1
                    print(f"[Line {i}] Invalid bounding box found in annotation file of {design_file}")
                    print(f"Removed: {widget} {x} {y} {w} {h}")
                    annotation_errors.append(i)
                    continue
                new_lines.append(line)
                if widget in widgets:
                    widgets[widget] += 1
                else:
                    errors += 1
                    print(f"[Line {i}] Unknown widget class {widget} found in annotation file of {design_file}")
            # NOTE Delete invalid annotations in label file
            f.seek(0)
            f.writelines(new_lines)
            f.truncate()
            del new_lines
        logger.report_scalar(title='Generator', series='total_widgets', value=sum(widgets.values()), iteration=i+1)
        logger.report_scalar(title='Generator', series='errors', value=errors, iteration=i+1)
        for widget in widgets:
            logger.report_scalar(title='Widget metrics', series=widget, value=widgets[widget], iteration=i+1)
    generated_files = len(files)
    env['generated_files'] = generated_files
    env['files'] = files

# Dataset helpers
def replace_class_names(files: list):
    """
    **Params:**
    - `files` List of tuples containing the image and annotation file paths

    Replaces the class names in the annotation files with a numerical representation.

    The number is the index of the class name in the global list of implemented types.
    This is necessary, as the YOLO training process cannot deal with string representations of classes.
    """
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
    """
    **Params:**
    - `input` List of tuples containing the image and annotation file paths
    - `split_ratio` Tuple of three values that sum up to 1 (train, val, test)

    Shuffles the provided input list and splits it into three parts based on the provided split ratio.
    The split ratio is a tuple of three values that sum up to 1.
    The first value is the ratio of the training set, the second value is the ratio of the validation set, and the third value is the ratio of the test set.
    """
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
    """
    **Params:**
    - `env` Environment dictionary containing the output folder and Micropython paths
    - `args` Arguments containing the type of generator and the split ratio

    Prepares and organizes a dataset from the generated files.

    The dataset is split into three parts based on the provided split ratio, otherwise a default split ratio of `(0.7, 0.1, 0.2)` is used.
    The dataset is organized into a folder structure that is compatible with the YOLO training process.
    Each dataset part (train, val, test) contains an image and a label folder, where the respective files will be moved after the shuffle split.
    The files will be renamed according to their destination folder and index in their partial list. (e.g. `[(train_0.jpg, train_0.txt), ...]`, `[(val_0.jpg, val_0.txt), ...]`, `[(test_0.jpg, test_0.txt), ...]`)

    A dataset YAML file is created that contains the path to the dataset, the folder structure, and the class indexes with their names.
    If the no_dataset flag is set, the dataset will be uploaded as a ZIP archive artifact to the ClearML task.
    Otherwise, another function will take care of dataset creation within ClearML by re-using all task information.
    """
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
    for folder in folders[1:]: # NOTE Skip first element to not delete the target_dir (might contain extra data/folders)
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    # Move files
    for i, (image, annotation) in enumerate(train_files):
        print(f"Moving {image} -> {train_img_folder}/train_{i}.jpg")
        shutil.move(image, os.path.join(train_img_folder, f"train_{i}.jpg"))
        shutil.move(annotation, os.path.join(train_label_folder, f"train_{i}.txt"))
    for i, (image, annotation) in enumerate(val_files):
        print(f"Moving {image} -> {val_img_folder}/val_{i}.jpg")
        shutil.move(image, os.path.join(val_img_folder, f"val_{i}.jpg"))
        shutil.move(annotation, os.path.join(val_label_folder, f"val_{i}.txt"))
    for i, (image, annotation) in enumerate(test_files):
        print(f"Moving {image} -> {test_img_folder}/test_{i}.jpg")
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
    """
    **Params:**
    - `env` Environment dictionary containing the output folder and Micropython paths
    - `args` Arguments containing the type of generator and the dataset name

    Creates a dataset within ClearML from the generated dataset files.

    The dataset is created with the provided dataset name and the static project name "LVGL UI Detector".
    The dataset creation re-uses the current task information and uploads the dataset files to ClearML.
    This ensures, that all created files, statistics and metadata are available in the linked task of the dataset.
    The dataset is given a final comment, stating the dataset name and its ID.

    After uploading, the dataset is finalized and cannot be modified anymore.
    """
    from clearml import Task, Dataset
    task = Task.current_task()
    if args.type == 'random':
        custom = f"(widgets={env['generated_widgets']},files={env['generated_files']})"
    else:
        custom = f"(files={env['generated_files']})"
    if not args.dataset:
        args.dataset = f"LVGL UI {custom}"
    else:
        args.dataset = f"{args.dataset} {custom}"
    task.rename(f"{args.type.upper()} UI Dataset")
    dataset = Dataset.create(dataset_name=args.dataset, dataset_project="LVGL UI Detector", use_current_task=True)
    dataset.add_files(env['dataset_folder'])
    dataset.upload()
    comment = f"Dataset '{args.dataset}' created: {dataset.id}"
    print(comment)
    dataset.set_metadata(env['metadata'])
    dataset.finalize()
    task.set_comment(comment)

# Environment helpers
def prepare_task(args: dict):
    """
    **Params:**
    - `args` Dictionary containing the CLI arguments

    Prepares a ClearML task, connecting the provided CLI arguments as task parameters.

    The task is initialized with the project name "LVGL UI Detector" and the task name "UI Generator".
    The task type is set to "data_processing" and custom tags are set based on the provided arguments (generator type, model-variant, etc.)
    """
    from clearml import Task
    tags = ["lvgl-ui-detector", args.type]
    if args.type == 'design':
        tags.append(args.variant)
    task = Task.init(project_name='LVGL UI Detector', task_name=f'UI Generator', task_type=Task.TaskTypes.data_processing, tags=tags)
    task.connect(args)

def prepare_environment(output_folder: str, mpy_path: str, mpy_main: str):
    """
    **Params:**
    - `output_folder` Path to the output folder
    - `mpy_path` Path to the Micropython binary
    - `mpy_main` Path to the main.py script of the LVGL UI Generator project
    
    **Returns:**
    - `dict` Environment dictionary containing the output folder, Micropython paths, and additional information

    Prepares the environment dictionary for the generation process.
    The target output folder is created and the paths to the Micropython generator and main script are stored.
    The environment dictionary is returned for further usage.
    This dictionary will further be extended by other functions in the generation process, to store additional information.
    """
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
    """
    **Params:**
    - `filepath` Path to the JSON file to load
    
    **Returns:**
    - `dict` Python dictionary containing the JSON file content

    Loads a JSON file from the provided file path and returns the content as a Python dictionary.
    """
    import json
    with open(filepath, 'r') as f:
        return json.load(f)

def verify_design_from_file(design_file: str, schema_file: str) -> tuple[bool, Exception]:
    """
    **Params:**
    - `design_file` Path to the design file to verify
    - `schema_file` Path to the JSON schema file to use for validation
    
    **Returns:**
    - `tuple[bool, Exception]` Tuple containing a boolean value and an exception (if any)

    Verifies the design file against the provided JSON schema file.
    The design file is loaded and validated against the schema, using the JSON schema library.
    If the design file is valid, the function returns True and None.
    If the design file is invalid, the function returns False and the exception that was raised during validation.
    """
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
    """
    **Params:**
    - `design` Design string to verify
    - `schema_file` Path to the JSON schema file to use for validation
    
    **Returns:**
    - `tuple[bool, Exception]` Tuple containing a boolean value and an exception (if any)

    Verifies the provided design string against the provided JSON schema file.
    The design string is loaded and validated against the schema, using the JSON schema library.
    If the design string is valid, the function returns True and None.
    If the design string is invalid, the function returns False and the exception that was raised during validation.
    """
    from jsonschema import validate
    from jsonschema.exceptions import ValidationError
    schema = load_json_file(schema_file)
    try:
        validate(instance=design, schema=schema)
        return True, None
    except ValidationError as e:
        return False, e

def count_design_widget_types(json_string, first: bool = True):
    """
    **Params:**
    - `json_string` JSON string containing the design to count the widget types from
    - `first` Flag indicating if this is the first call of the function
    
    **Returns:**
    - `dict` Dictionary containing the widget types and their occurrences

    Counts the occurrences of each widget type in the provided JSON string.

    The JSON string is loaded and parsed into a Python dictionary.
    The function iterates through each widget and counts the occurrences of each widget type.
    If a widget is a container, the layout type is also counted.
    The function is recursive and counts the widget types in the children of each widget.
    The function returns a dictionary with the widget types as keys and their occurrences as values.
    """
    import json
    # Load the JSON string into a Python dictionary
    data = json.loads(json_string)
    # Create a dictionary to count occurrences of each widget type
    type_count = {}
    if first:
        type_count[data['ui']['root']['type']] = 1
        widgets = data['ui']['root']['children']
    else:
        # If this is a recursive call, the 'children' list is directly passed
        widgets = data
    # Iterate through each widget and increment its type count in the dictionary
    for widget in widgets:
        widget_type = widget['type']
        if widget_type == 'container':
            type_count[widget['options']['layout_type']] = type_count.get(widget['options']['layout_type'], 0) + 1
        type_count[widget_type] = type_count.get(widget_type, 0) + 1
        if 'children' in widget:
            # Recursively count the widget types in the children of this widget
            child_type_count = count_design_widget_types(json.dumps(widget['children']), False)
            # Merge the child type counts into the main type count dictionary
            for child_type in child_type_count:
                type_count[child_type] = type_count.get(child_type, 0) + child_type_count[child_type]
    # Return the dictionary of widget type counts
    return type_count

def ask_gpt(openai: dict):
    """
    **Params:**
    - `openai` Configuration dictionary for the OpenAI GPT API
    
    **Returns:**
    - `dict` Response dictionary from the OpenAI GPT API

    Calls the OpenAI GPT API with the provided configuration.

    The OpenAI API is used to generate a response based on the provided configuration.
    The response is returned to the caller.
    Used configurations are:
    - model: The model variant to use for the response generation.
    - messages: The messages to provide to the model for the response generation.
    - max_tokens: The maximum amount of tokens to generate in the response.
    - format: The format of the response to generate.
    - stop: The stop condition for the response generation. (Optional)

    You may either use temperature or top_p for response generation, but not both as it is mutually exclusive.
    This is the recommended pattern by the OpenAI API.
    If neither are provided, the response generation will be done using the default temperature and top_p values, as stated in the OpenAI API documentation.
    """
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
    """
    **Params:**
    - `openai` Configuration dictionary for the OpenAI GPT API
    
    **Returns:**
    - `dict` Response from the OpenAI GPT API

    Executes the `ask_gpt` function to generate a design idea, with the static system prompt for proper structure of the design idea output.

    The OpenAI API is used to generate a response based on the provided configuration.
    A random chosen value from the combinated list of themes and topics is used as user prompt for the design idea.
    A chosen value is then removed from the list to avoid repetition.
    The list contains 10000 unique combinations of UI contexts and themes.
    The system prompt is designed, so that the design idea will follow a specific structure and contain necessary information in regards to the LVGL UI generator.
    The response is returned to the caller.
    """
    import random
    from datetime import datetime
    random.seed(datetime.now().timestamp())
    # NOTE Shuffle themes and topics to randomize selection
    random.shuffle(combinations)
    theme_topic = random.choice(combinations)
    # NOTE Remove the used theme and topic to avoid repetition
    combinations.remove(theme_topic)
    theme, topic = theme_topic
    system_message = """
You are a UI designer.
Your goal is to create a new design idea and guideline for an user interface to be given to developers.
The user will ask for a new design to be created.
You will imagine a new user interface and provide a detailed description of the design.
Make sure to include the following in your design idea:
- A title for the design
- The context of the design
- The visual theme of the design
- A list of style groups
- A list of used widgets, including their purpose and style
- A high-level description of the design

You must adhere to these constraints and rules.

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
- Roller: A special dropdown with a rolling interface for selecting entries
- Scale: A scale for displaying a range of values
- Slider: A slider for selecting a value within a range
- Spinbox: Spinbox widget
- Switch: A switch for toggling between two states
- Table: A tabular widget for displaying data
- TextArea: A field for input or display of multiple text lines
- Container: A special widget containing other widgets in a structured layout

Design elements can be structurally combined using the special widget 'container'.
These special widgets must have an 'id' attribute to be referenced by other widgets.
The structure of such a container may be defined as a layout of the following types and purposes:
- none: Widgets are structured using absolute positioning (Widgets placed in the container must be positioned using a specific style group that defines their X and Y coordinates)
- flex: Widgets are structured in a flexible layout that adjusts to the available space horizontally or vertically (using flex_flow: row, column, row_wrap, column_wrap, row_reverse, column_reverse)
Widget sizes must avoid overlap with other widgets and can be controlled by setting width and height in a style group.

Design elements can be styled individually or identically using a style group.
Style groups define a set of style properties that are applied to the widgets that use them.
The style properties include color, size, padding, margin, opacity, border, lines, shadows and more.
The style groups are named using a unique identifier.
A widget may have multiple styles applied to it, and a style group may be applied to multiple widgets.

ALLOWED STYLE PROPERTIES:
These properties are applicable to style groups and can be used to define the appearance of widgets.
- bg_color: Background color of the widget
- bg_opa: Opacity of the background color of the widget
- border_color: Color of the border of the widget
- border_opa: Opacity of the border of the widget
- border_width: Width of the border of the widget
- outline_width: Width of the outline of the widget
- outline_color: Color of the outline of the widget
- outline_opa: Opacity of the outline of the widget
- shadow_width: Width of the shadow of the widget
- shadow_offset_x: Horizontal offset of the shadow of the widget
- shadow_offset_y: Vertical offset of the shadow of the widget
- shadow_color: Color of the shadow of the widget
- shadow_opa: Opacity of the shadow of the widget
- line_width: Width of the line of the widget
- line_dash_width: Width of the dashes of the line of the widget
- line_dash_gap: Gap between the dashes of the line of the widget
- line_rounded: A boolean representing if the line of the widget is rounded
- line_color: Color of the line of the widget
- line_opa: Opacity of the line of the widget
- arc_width: Width of the arc widget
- arc_color: Color of the arc widget
- arc_opa: Opacity of the arc widget
- arc_rounded: A boolean representing if the arc widget is rounded
- text_color: Color of the text of the widget
- text_opa: Opacity of the text of the widget
- text_letter_space: Letter space of the text of the widget
- text_line_space: Line space of the text of the widget
- opa: Opacity of the widget itself
- align: Alignment of the widget
- x: X position of the widget
- y: Y position of the widget
- min_width: Minimum width of the widget
- min_height: Minimum height of the widget
- max_width: Maximum width of the widget
- max_height: Maximum height of the widget
- length: Length of the widget
- pad_all: Padding of the widget on all sides
- pad_hor: Horizontal padding of the widget
- pad_ver: Vertical padding of the widget
- pad_gap: Gap between the padding of the widget
- pad_top: Top padding of the widget
- pad_bottom: Bottom padding of the widget
- pad_left: Left padding of the widget
- pad_right: Right padding of the widget
- pad_row: Padding of the widget on the row
- pad_column: Padding of the widget on the column
- margin_top: Top margin of the widget
- margin_bottom: Bottom margin of the widget
- margin_left: Left margin of the widget
- margin_right: Right margin of the widget

CONSTRAINT:
- A design must always be a visually complete representation of a single window, ignoring any possible interactions of the user interface.
Think of the design as a static image that represents the final appearance of the user interface in a single frame.
- The size of the window must be 640 x 640 pixels.
- Font is not a design element, so you do not need to specify font styles or sizes.
- The design idea is targeting the LVGL library

TASK:
It is your task to create a design idea for a user interface that adheres to these design rules and constraints.
You will be given a random topic and theme to base your design idea on.
Try to incorporate as many varied widgets as possible in the design.
"""
# NOTE Removed the following constraints to simplify the task
# """
# - grid: Widgets are structured in rows and columns, each cell containing a widget
# """
    initial_prompt = f"""
Create a new UI design about '{topic}' in the theme of '{theme}'.
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
    """
    **Params:**
    - `openai` Configuration dictionary for the OpenAI GPT API
    - `design` Design dictionary containing the design idea and previous validation errors
    
    **Returns:**
    - `tuple[bool, str, Exception]` Tuple containing a boolean value, the generated JSON design and an exception (if any)

    Executes the `ask_gpt` function to generate a JSON design from the provided design idea.

    The OpenAI API is used to generate a response based on the provided configuration.
    The design idea is provided as user input to the model for the response generation.
    The system prompt is designed, so that the output JSON will follow special rules and constraints for the LVGL UI generator.
    Additionally, the design schema is added to the system prompt, as formal specification guidance for the JSON output.
    The JSON response is validated against the schema after generation.
    If the provided design dictionary contains validation errors from previous attempts, the last error is provided as additional user prompt for GPT to correct.
    The response is returned to the caller for further processing.
    """
    import os, json
    schema_file = os.path.join(os.path.curdir, 'schema', 'design_file.schema.json')
    openai['schema'] = json.load(open(schema_file, 'r'))
    system_message = """
You are a UI generator.
Your goal is to create a new single window UI using a specialized JSON format.
The format specification is available in the design.schema.json file below.
Follow the provided design guideline of the user when replicating the design idea using the structure of the JSON format.
Always output a valid JSON object that represents the UI design.

You ALSO MUST adhere to the following SPECIAL RULES:
- Never use 'grid' container. Use either 'none' or 'flex' container.
- Containers must be used to set style for the whole window or the specific group of widgets (such as background color and other general styles).
- Each widget placed inside a 'none' container must have an associated style defining its X and Y coordinates.
- Widgets may have multiple styles applied to them
- Window size MUST be 640 x 640 pixels
- You MUST make sure that coordinates of widgets do not overlap due to width and height of the widgets
- You MUST make sure that the widgets are within the window bounds (0, 0, 640, 640)
    """
# NOTE Removing the use of grid containers, since they are too complicated to get a good looking result
# '''
# You ALSO MUST adhere to the following SPECIAL RULES:
# - If a widget is under the 'children' property of a container AND the container is of layout_type 'grid', then those children must have the 'placement' property.
# This also applies to nested containers within the 'children' property of a container.
# This placement property must adhere to the following schema:
# "placement": {
#     "type": "object",
#     "properties": {
#         "col_pos": { "type": "integer" },
#         "col_span": { "type": "integer" },
#         "row_pos": { "type": "integer" },
#         "row_span": { "type": "integer" }
#     },
#     "required": ["col_pos", "col_span", "row_pos", "row_span"]
# }
# - The window size defined under the 'window' object MUST be square. (equal width and height)
# - Container widgets must have an associated 'style' which defines their HEIGHT and WIDTH.
# '''
    system_message += f"\ndesign.schema.json:\n{str(openai['schema'])}"
    initial_prompt = "Create a new user interface for this design idea:\n"
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
    json_file = os.path.join(env['design_folder'], f'attempts', f"design_{design['iteration']}_{design['attempts']}.json")
    with open(json_file, 'w') as f:
        f.write(json_design)
    # Check if response is valid JSON
    is_valid, error = verify_design_from_file(json_file, schema_file)
    return is_valid, json_design, error

def generate_designs(env: dict, args: argparse.Namespace):
    """
    **Params:**
    - `env` Environment dictionary containing the output folder and Micropython paths
    - `args` Arguments containing the design type, model variant, and design count

    Generates UI designs based on the provided configuration and arguments.
    The OpenAI API is used to generate design ideas and JSON designs.

    **First**, a design idea is generated using the `ask_gpt_for_design_idea` function.
    The LLM is prompted with a randomly chosen combination of the `themes` and `topics`  to generate a design idea.
    Once chosen, the theme and topic are removed from the `combinations` list to avoid repetition.
    The design idea is saved to a markdown file for reference in the created `attempts` folder.

    **Then**, a JSON design is generated using the `ask_gpt_for_design_json` function.
    The created JSON design is saved as an attempt file for reference.
    The JSON file is then validated against the design schema to ensure its correctness.
    If it contains errors, the validation error is saved to a file for reference.
    Any validation errors are looped back to another attempt to generate a valid JSON design.
    Any design idea will be attempted to generate a valid JSON design up to 3 times.
    This limit is set to avoid infinite costs incurred through the OpenAI API usage.

    The widget types of the generated designs are counted and reported to the ClearML task.
    The total widget count is reported as a histogram, showing the distribution of widget types in the generated designs.
    All valid design files are saved to the output folder for further processing and usage in the dataset generation.
    """
    import os
    from clearml import Task
    import numpy as np
    # TODO should connect stuff to ClearML task
    task = Task.current_task()
    task.add_tags([args.variant, args.model])
    logger = task.get_logger()
    open_ai = {}
    env['design_folder'] = os.path.join(args.output_folder, args.type, 'designs')
    os.makedirs(env['design_folder'], exist_ok=True)
    os.makedirs(os.path.join(env['design_folder'], 'attempts'), exist_ok=True)
    open_ai['model'] = args.model
    open_ai['max_tokens'] = args.max_tokens
    if hasattr(args, 'temperature'):
        open_ai['temperature'] = args.temperature
    elif hasattr(args, 'top_p'):
        open_ai['top_p'] = args.top_p
    total_widget_count = {}
    open_ai['valid_designs'] = 0
    for i in range(args.designs):
        print(f"Generating design {i}...")
        valid_design = False
        design = {}
        design['attempts'] = 0
        design['errors'] = []
        design['iteration'] = i
        design['idea'] = ask_gpt_for_design_idea(open_ai)
        with open(os.path.join(env['design_folder'], f'attempts', f"design_{i}_idea.md"), 'w') as f:
            f.write(design['idea'])
        while not valid_design and design['attempts'] < 3: # NOTE Limiting attempts (to avoid infinite costs for now)
            valid_design, design['json_raw'], error = ask_gpt_for_design_json(open_ai, design)
            if error:
                design['errors'].append(error)
                print(f"JSON generation failed with error.")
                with open(os.path.join(env['design_folder'], f'attempts', f"design_{i}_errors.txt"), 'a') as f:
                    f.write(f"\nAttempt {design['attempts']}:\n")
                    f.write(str(error))
                design['attempts'] += 1
        else:
            logger.report_scalar(title='GPT Designer', series='errors', value=len(design['errors']), iteration=i+1)
            if valid_design:
                print(f"Generated valid design after {design['attempts'] + 1} attempts")
                open_ai['valid_designs'] += 1
            else:
                print(f"Failed to generate valid design after {design['attempts'] + 1} attempts")
                continue
        # print(f"Design JSON:\n{design['json_raw']}")
        design['file'] = os.path.join(env['design_folder'], f"design_{i}.json")
        with open(design['file'], 'w') as f:
            f.write(design['json_raw'])
        # Count widget types
        widget_count = count_design_widget_types(design['json_raw'])
        logger.report_scalar(title='GPT Designer', series='total_widgets', value=sum(widget_count.values()), iteration=i+1)
        for widget in widget_count:
            logger.report_scalar(title='GPT Designer', series=widget, value=widget_count[widget], iteration=i+1)
            if widget not in total_widget_count:
                total_widget_count[widget] = widget_count[widget]
            else:
                total_widget_count[widget] += widget_count[widget]
        open_ai[f'design_{i}'] = design
    env['OPEN_AI'] = open_ai
    generated_widgets = sum(total_widget_count.values())
    generated_designs = open_ai['valid_designs']
    histogram_values = np.array([[generated_widgets], [generated_designs]] + [[value] for value in total_widget_count.values()])
    histogram_labels = ['Widgets', 'Designs'] + [key for key in total_widget_count.keys()]
    logger.report_histogram(title='Generated', series='total', values=histogram_values, labels=histogram_labels, yaxis='Count')

# CLI helpers (used for argparse to show help for subparsers)
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
    """
    Creates the CLI parser for the UI Detector dataset generation script.
    The parser includes arguments for the output folder, Micropython paths, dataset name, split ratio and dataset creation.
    Additionally, subparsers are added for the different UI generation types (random, design) and their respective arguments.
    """
    parser = argparse.ArgumentParser(description='Generate UI Detector dataset', add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-h', '--help', action=_HelpAction, help='show this help message and exit') # add custom help
    parser.add_argument('-o', '--output-folder', type=str, default='tmp/output', help='Output folder')
    parser.add_argument('--mpy-path', type=str, default='', help='Path to MicroPython binary (loads from environment MICROPYTHON_BIN if not provided)')
    parser.add_argument('--mpy-main', type=str, default='', help='Path to main.py of micropython script (loads from environment MICROPYTHON_MAIN if not provided)')
    parser.add_argument('-d', '--dataset', type=str, default='', help='Custom name of the dataset written in the task comment')
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
    gpt_arg = design_gpt.add_mutually_exclusive_group()
    gpt_arg.add_argument('--temperature', type=float, default=0.7, help='ChatGPT sampling temperature')
    gpt_arg.add_argument('--top-p', type=float, default=1.0, help='ChatGPT top-p sampling')
    return parser

def validate_cli_args(args: dict):
    """
    **Params:**
    - `args` Dictionary containing the CLI arguments

    Validates the provided CLI arguments and environment variables.

    The function checks if a micropython path and script were provided. If not, it checks for existance in the environment variables.
    If the paths are not provided or do not exist, the function prints an error message and exits the script.
    Additionally, the function checks for the design folder in case of a local design generator and prints an error if it is not provided.
    If the GPT design generator is used, the function checks for the API key and model name and prints an error if they are not provided.
    """
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
    env = prepare_environment(args.output_folder, args.mpy_path, args.mpy_main)
    if args.type == 'random':
        capture_random(env, args)
    elif args.type == 'design':
        if args.variant == 'local':
            capture_design(args.design_folder)
        elif args.variant == 'remote':
            pass
        elif args.variant == 'gpt':
            try:
                generate_designs(env, args)
            except KeyboardInterrupt as e:
                print("Aborted design generation.")
            capture_design(env['design_folder'])
    prepare_dataset(env, args)
    if not args.no_dataset:
        create_dataset(env, args)