from ray import tune
import os
from main import train_structured, evaluate
from tune_teacher import train_model_tune
from structure import collect_pooled_neighborhood
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from functools import partial
import torch
import scipy.stats as stats
import pandas as pd


storage_path = os.getcwd() + "/results"
exp_name = "cora_fixed_teacher_params"
data = PubmedGraphDataset()

experiment_path = os.path.join(storage_path, exp_name)
print(experiment_path)
print(f"Loading results from {experiment_path}...")


# load and p
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g = data[0]
g = g.int().to(device)
features = g.ndata["feat"]
labels = g.ndata["label"]
masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
classes = data.num_classes

nbhds = {
    'train': collect_pooled_neighborhood(g, 1, masks[0]),
    'val': collect_pooled_neighborhood(g, 1, masks[1]),
}

#train_joint_KL_other(g, features, labels, masks, model2, modelMLP2, 0.5, 0.5)
#train_structured(g, features, labels, masks, model, modelMLP, nbhds, configs['train_params'])
#    train_MLP(g, features, labels, masks, modelMLP)

run = partial(train_structured, g, features, labels, masks, classes, nbhds)

restored_tuner = tune.Tuner.restore(experiment_path, trainable=run)
result_grid = restored_tuner.get_results()
best_result = result_grid.get_best_result("accS", "max")
#best_result.metrics_dataframe.plot("training_iteration", "accS")

result_df = result_grid.get_dataframe()
result_df = result_df[result_df["accS"] > 0.5].sort_values(by=['accS'])


#print(result_df[["config/lr", "config/decay", "val_loss", "acc"]])

print(result_df[["config/s_lr","config/s_decay", "config/s_kl", "epoch","val_loss", "accS"]])
(func pid=1503786) Checkpoint successfully created at: Checkpoint(fi




