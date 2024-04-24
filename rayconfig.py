from ray import train, tune
from ray.train._internal.session import TrialInfo
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper, ExperimentPlateauStopper, CombinedStopper, MaximumIterationStopper


def setup_tuning(run, param_space, storage_path, args):
    trial = TrialPlateauStopper(metric="acc", std=0.01, metric_threshold=0.5, mode="max")
   # exp = ExperimentPlateauStopper(metric="acc", std=0.01, top=20, patience=5)
    maxiter = MaximumIterationStopper(int(args.stopping))
    stopper = CombinedStopper(trial, maxiter)
    scheduler = ASHAScheduler(metric="val_loss", mode="min", grace_period=15)
    tuner = tune.Tuner(run,
                       param_space=param_space,
                       tune_config=tune.TuneConfig(
                           scheduler=scheduler,
                           num_samples=int(args.samples),
                       ),
                       run_config=train.RunConfig(
                           storage_path=storage_path,
                           name=args.name,
                           stop=stopper,
                       )
    )
    return tuner

def setup_training(run, param_space, storage_path, args):
    trial = TrialPlateauStopper(metric="loss", std=0.001, metric_threshold=0.5, mode="max")
    maxiter = MaximumIterationStopper(int(args.stopping))
    stopper = CombinedStopper(trial, maxiter)
    tuner = tune.Tuner(run,
                       param_space=param_space,
                       tune_config=tune.TuneConfig(
                           num_samples=10,
                       ),
                       run_config=train.RunConfig(
                           storage_path=storage_path,
                           name=args.name,
                           stop=stopper,
                       )
    )
    return tuner
