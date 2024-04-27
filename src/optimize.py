# HPO helpers
def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print(f'New top performing job found!\nJob ID: {job_id}, mAP: {objective_value}, iteration: {objective_iteration}')

def optimize_hyperparameters(template_id:str, run_local:bool = False):
    from clearml.automation import UniformParameterRange, DiscreteParameterRange
    from clearml.automation import HyperParameterOptimizer
    from clearml.automation.optuna import OptimizerOptuna
    from clearml import Task
    print("Optimizing hyperparameters")
    task = Task.init(
        project_name='Hyper-Parameter Optimization (UI Detector)',
        task_name='Automatic Hyper-Parameter Optimization (UI Detector)',
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )
    # job = ClearmlJob(
    #     base_task_id=template_id,
    #     parameter_override={'data': ''},
    # )
    an_optimizer = HyperParameterOptimizer(
        # This is the experiment we want to optimize
        base_task_id=template_id,
        hyper_parameters=[
            # Other hyperparameters we want to optimize
            DiscreteParameterRange('General/batch', values=[16,32,64,128,256]),
            DiscreteParameterRange('General/epochs', values=[25,50,75,100,125,150]),
            UniformParameterRange('General/lr0', min_value=0.001, max_value=0.1),
            UniformParameterRange('General/momentum', min_value=0.85, max_value=0.95),
            UniformParameterRange('General/weight_decay', min_value=0.0001, max_value=0.001),
            DiscreteParameterRange('General/imgsz', values=[320, 480, 640]),
            DiscreteParameterRange('General/warmup_epochs', values=[1, 3, 5])
        ],
        objective_metric_title='val',
        objective_metric_series='metrics/mAP50-95(B)',
        objective_metric_sign='max',
        optimizer_class=OptimizerOptuna,
        execution_queue='training',
        pool_period_min=5,
        max_iteration_per_job=30,
        total_max_jobs=1,
        max_number_of_concurrent_tasks=1
    )
    # experiment template to optimize in the hyperparameter optimization
    args = {
        'template_task_id': id if id else '',
        'run_as_service': False,
    }
    args = task.connect(args)

    # report every 12 seconds, this is way too often, but we are testing here J
    an_optimizer.set_report_period(2)
    # start the optimization process, callback function to be called every time an experiment is completed
    if run_local:
        an_optimizer.start_locally(job_complete_callback=job_complete_callback)
    else:
        an_optimizer.start(job_complete_callback=job_complete_callback)
    # set the time limit for the optimization process (2 hours)
    an_optimizer.set_time_limit(in_minutes=2*60)
    an_optimizer.wait()

if __name__ == "__main__":
    import argparse
    from clearml import Task
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for a ClearML training task")
    parser.add_argument("--id", type=str, help="Task ID to optimize")
    parser.add_argument("--local", action="store_true", help="Run the optimization locally")
    args = parser.parse_args()
    print(f"Task to optimize: {args.id}")
    template = Task.get_task(task_id=args.id)
    if template is None:
        raise ValueError("Task not found.")
    # Hyperparameter optimization
    optimize_hyperparameters(template.id, args.local)