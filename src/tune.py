def yolo_tune(args):
    from ultralytics import YOLO
    from clearml import Task
    # Setup task
    task = Task.init(project_name='UI Detector YOLO tune', task_name='Tune', task_type=Task.TaskTypes.optimizer)
    task.connect(args)

    # Initialize the YOLO model
    model = YOLO(args.model)

    # Tune hyperparameters on COCO8 for 30 epochs
    model.tune(data=args.data, epochs=args.epochs, iterations=args.iterations, optimizer=args.optimizer)


if __name__ == "__main__":
    # args: model: str, data: str, epochs: int, iterations: int, optimizer: str
    from ultralytics import YOLO
    import argparse
    parser = argparse.ArgumentParser(description='Tune dataset hyperparameters for a YOLO model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, required=True, help=f'Model name (from: {', '.join(YOLO.available_models())})')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML file')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations per epoch')
    parser.add_argument('--optimizer', type=str, default='adam', help='YOLO Optimizer (from: adam, sgd)')
    args = parser.parse_args()
    yolo_tune(args)
