"""
This script is used to optimize hyperparameters for a ClearML training task.
The task ID to optimize is provided as an argument.
The optimization process is logged to a new ClearML task.
Any individual training tasks created during the optimization process are logged as child tasks.

The script may be called according to the following usage information:

    usage: optimize.py [-h] [--id ID] [--local] [--pool-period POOL_PERIOD] [--max-jobs MAX_JOBS] [--max-concurrent MAX_CONCURRENT] [--max-iterations MAX_ITERATIONS]
                    [--time-limit TIME_LIMIT] [--top-k TOP_K] [--execution-queue EXECUTION_QUEUE]

    Optimize hyperparameters for a ClearML training task

    options:
    -h, --help            show this help message and exit
    --id ID               Task ID to optimize (default: None)
    --local               Run the optimization locally (default: False)
    --pool-period POOL_PERIOD
                            Pool period in minutes (default: 5)
    --max-jobs MAX_JOBS   Maximum number of jobs to run (default: 25)
    --max-concurrent MAX_CONCURRENT
                            Maximum number of concurrent tasks (default: 2)
    --max-iterations MAX_ITERATIONS
                            Maximum number of iterations per job (default: 100)
    --time-limit TIME_LIMIT
                            Time limit for optimization in minutes (default: 120)
    --top-k TOP_K         Number of top experiments to print (default: 5)
    --execution-queue EXECUTION_QUEUE
                            Execution queue for optimization (default: training
"""

# HPO helpers
def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    """
    **Params:**
    - `job_id` The ID of the job that was completed.
    - `objective_value` The value of the objective metric.
    - `objective_iteration` The iteration at which the objective metric was achieved.
    - `job_parameters` The parameters of the job that was completed.
    - `top_performance_job_id` The ID of the job with the top performance.

    Callback function to be called every time an experiment is completed.
    """
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print(f'New top performing job found!\nJob ID: {job_id}, mAP: {objective_value}, iteration: {objective_iteration}')

def optimize_hyperparameters(args: dict):
    """
    **Params:**
    - `args` Arguments for the optimization process.

    Optimize hyperparameters for a ClearML training task.
    The task ID to optimize is provided in the args dictionary.
    The optimization process is logged to a new ClearML task.
    Any individual training tasks created during the optimization process are logged as child tasks.
    The parameter optimization is performed using the Optuna optimizer.

    The used hyperparameters and ranges are statically defined:
    - `General/batch`: `[16, 32, 64, 128, 256]`
    - `General/epochs`: `[50, 75, 100]`
    - `General/lr0`: `[0.001, 0.1]` (step size: `0.003`)
    - `General/momentum`: `[0.85, 0.95]` (step size: `0.02`)
    - `General/weight_decay`: `[0.0001, 0.001]` (step size: `0.0002`)
    - `General/imgsz`: `[320, 480, 640]`
    - `General/warmup_epochs`: `[1, 3, 5]`
    """
    from clearml.automation import UniformParameterRange, DiscreteParameterRange
    from clearml.automation import HyperParameterOptimizer
    from clearml.automation.optuna import OptimizerOptuna
    from clearml import Task
    print("Optimizing hyperparameters")
    task = Task.init(
        project_name='Hyper-Parameter Optimization (UI Detector)',
        task_name='Automatic HPO (UI Detector)',
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )
    task.connect(args)
    hyper_parameters = [
        DiscreteParameterRange('General/batch', values=[16,32,64,128,256]),
        DiscreteParameterRange('General/epochs', values=[50,75,100]),
        UniformParameterRange('General/lr0', min_value=0.001, max_value=0.1, step_size=0.003),
        UniformParameterRange('General/momentum', min_value=0.85, max_value=0.95, step_size=0.02),
        UniformParameterRange('General/weight_decay', min_value=0.0001, max_value=0.001, step_size=0.0002),
        DiscreteParameterRange('General/imgsz', values=[320, 480, 640]),
        DiscreteParameterRange('General/warmup_epochs', values=[1, 3, 5])
    ]
    an_optimizer = HyperParameterOptimizer(
        # This is the experiment we want to optimize
        base_task_id=args.id,
        hyper_parameters=hyper_parameters,
        objective_metric_title='val',
        objective_metric_series='metrics/mAP50-95(B)',
        objective_metric_sign='max',
        optimizer_class=OptimizerOptuna,
        execution_queue=args.execution_queue,
        pool_period_min=args.pool_period,
        total_max_jobs=args.max_jobs,
        max_number_of_concurrent_tasks=args.max_concurrent,
        max_iteration_per_job=args.max_iterations,
    )
    # experiment template to optimize in the hyperparameter optimization
    args = {
        'template_task_id': id if id else '',
        'run_as_service': False,
    }
    args = task.connect(args)

    an_optimizer.set_report_period(5)
    # start the optimization process, callback function to be called every time an experiment is completed
    if hasattr(args, 'local') and args.local:
        an_optimizer.start_locally(job_complete_callback=job_complete_callback)
    else:
        an_optimizer.start(job_complete_callback=job_complete_callback)
    # set the time limit for the optimization process
    an_optimizer.set_time_limit(in_minutes=6*60)
    an_optimizer.wait()
    # optimization is completed, print the top performing experiments id
    top_exp = an_optimizer.get_top_experiments(top_k=args.top_k)
    print([t.id for t in top_exp])
    # make sure background optimization stopped
    an_optimizer.stop()
    print('We are done, good bye!')

if __name__ == "__main__":
    import argparse
    from clearml import Task
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for a ClearML training task", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--id", type=str, help="Task ID to optimize")
    parser.add_argument("--local", action="store_true", help="Run the optimization locally")
    parser.add_argument("--pool-period", type=int, default=5, help="Pool period in minutes")
    parser.add_argument("--max-jobs", type=int, default=25, help="Maximum number of jobs to run")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Maximum number of concurrent tasks")
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum number of iterations per job")
    parser.add_argument("--time-limit", type=int, default=2*60, help="Time limit for optimization in minutes")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top experiments to print")
    parser.add_argument("--execution-queue", type=str, default="training", help="Execution queue for optimization")
    args = parser.parse_args()
    print(f"Task to optimize: {args.id}")
    template = Task.get_task(task_id=args.id)
    if template is None:
        raise ValueError("Task not found.")
    # Hyperparameter optimization
    optimize_hyperparameters(args)
