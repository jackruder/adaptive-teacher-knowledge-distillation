from ray import tune
from utils import *
import torch

grid_log = [1e-7,5e-7,1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,0.1]

single_model_search = {
    "decay": tune.loguniform(1e-8, 1e-2),
    "lr": tune.loguniform(1e-7,5e-1),
}

glnn_cfg = {
    's_kl': tune.uniform(0, 0.5), # we know low values work well, so cap the range to 0.5
    "s_decay": tune.loguniform(1e-8, 1e-2),
    "s_lr": tune.loguniform(1e-6,5e-1),
    "s_lsp": 0, # dont use structure loss
    'pt_ratio': 3, # use fully trained teacher
}

lsp_fixed_cfg = glnn_cfg.copy()
lsp_fixed_cfg["s_lsp"] = tune.loguniform(0.1,100) #
lsp_fixed_cfg["kernel"] = tune.choice(["l2","rbf","linear"]) #
lsp_fixed_cfg["sigma"] = tune.loguniform(0.1,1000) # for rbf kernel


atkd_no_lsp_cfg = glnn_cfg.copy()
atkd_no_lsp_cfg['t_kl']: tune.uniform(0,1) ## 1
atkd_no_lsp_cfg['t_lr']: tune.loguniform(1e-8, 1e-2) ## 1
atkd_no_lsp_cfg['t_decay']: tune.loguniform(1e-8, 1e-2)
atkd_no_lsp_cfg['pt_ratio']: tune.choice([0,1,2,3]) ## 0%, 33%, 66%, 100%, resp


atkd_lsp_cfg = atkd_no_lsp_cfg.copy()
atkd_lsp_cfg["s_lsp"] = tune.loguniform(0.1,100) #
atkd_lsp_cfg["kernel"] = tune.choice(["l2","rbf","linear"]) #
atkd_lsp_cfg["sigma"] = tune.loguniform(0.1,1000) # for rbf kernel

cora_find_student_params_cfg = {
    't_kl': tune.choice([1, 0.7]), ## 1
    's_kl': tune.uniform(0, 0.3), # 0.1
    "t_decay": tune.loguniform(1e-8, 1e-3),
    "s_decay": tune.loguniform(1e-8, 1e-3),
    "t_lr": tune.loguniform(1e-6,1e-3),
    "s_lr": tune.loguniform(5e-4,5e-1),
    "nbhd_size": 1,
    'kernel': torch.dot,
    'pt_epochs': 50,
    "pt_lr": 1e-3,
}

citeseer_find_student_params_cfg = {
    't_kl': tune.uniform(0.3,1), ## 1
    's_kl': tune.uniform(0, 0.3), # 0.1
    "t_decay": 1e-4,
    "s_decay": tune.loguniform(1e-8, 1e-3),
    "t_lr": tune.loguniform(1e-7,5e-4),
    "s_lr": tune.loguniform(1e-3,2e-2),
    "nbhd_size": 1,
    'kernel': torch.dot,
    'pt_epochs': 80,
    "pt_lr": 1e-2,
    "s_lsp": 0,
}

pubmed_find_student_params_cfg = {
    't_kl': tune.uniform(0.,1), ## 1
    's_kl': tune.uniform(0, 0.5), # 0.1
    "t_decay": 5e-4,
    "s_decay": tune.loguniform(1e-8, 1e-3),
    "t_lr": tune.loguniform(1e-7,5e-4),
    "s_lr": tune.loguniform(1e-3,2e-2),
    "nbhd_size": 1,
    'kernel': torch.dot,
    'pt_epochs': 10,
    "pt_lr": 4e-2,
    "s_lsp": 0,
}

cora_find_teacher_params_cfg = {
    't_kl': tune.uniform(0.7,1), ## 1
    's_kl': 0.1,
    "t_decay": 5e-5,#tune.loguniform(5e-6, 5e-4),
    "s_decay": 5e-4,
    "t_lr": tune.loguniform(1e-7,5e-4),
    "s_lr": 0.03,
    "nbhd_size": 1,
    'kernel': torch.dot,
    'pt_epochs': tune.choice([0, 15, 35, 50]),
    "pt_lr": 1e-3,
    "s_lsp": 0,
}

citeseer_find_teacher_params_cfg = {
    't_kl': tune.uniform(0.5,1),
    's_kl': 0.3,
    "t_decay": 1e-4,
    "s_decay": 1e-4,
    "t_lr": tune.loguniform(1e-7,1e-2),
    "s_lr": 0.01,
    "nbhd_size": 1,
    'kernel': torch.dot,
    'pt_epochs': 4,#tune.choice([0,4,8,13]),
    "pt_lr": 1e-2,
    "s_lsp": 0,
}

pubmed_find_teacher_params_cfg = {
    't_kl': tune.uniform(0,0.5),
    's_kl': 0.,
    "t_decay": 5e-4,
    "s_decay": 1e-4,
    "t_lr": tune.loguniform(1e-6,1e-2),
    "s_lr": 0.015,
    "nbhd_size": 1,
    'kernel': torch.dot,
    'pt_epochs': 8,#,tune.choice([0,4,8,12]),
    "pt_lr": 4e-2,
    "s_lsp": 0,
}


best_cora_find_teacher_params_cfg = {
    't_kl': 0.8, ## 1
    's_kl': 0.1,
    "t_decay": 5e-5,#tune.loguniform(5e-6, 5e-4),
    "s_decay": 5e-4,
    "t_lr": 4e-4,
    "s_lr": 0.03,
    "nbhd_size": 1,
    'kernel': torch.dot,
    'pt_epochs': 35,
    "pt_lr": 1e-3,
    "s_lsp": 0,
}

best_citeseer_find_teacher_params_cfg = {
    't_kl': 0.5,
    's_kl': 0.3,
    "t_decay": 1e-4,
    "s_decay": 1e-4,
    "t_lr": 3e-6,
    "s_lr": 0.01,
    "nbhd_size": 1,
    'kernel': torch.dot,
    'pt_epochs': 4,#tune.choice([0,4,8,13]),
    "pt_lr": 1e-2,
    "s_lsp": 0,
}

best_pubmed_find_teacher_params_cfg = {
    't_kl': 0.2,
    's_kl': 0.0,
    "t_decay": 5e-4,
    "s_decay": 1e-4,
    "t_lr": 2e-6,
    "s_lr": 0.015,
    "nbhd_size": 1,
    'kernel': torch.dot,
    'pt_epochs': 8,#,tune.choice([0,4,8,12]),
    "pt_lr": 4e-2,
    "s_lsp": 0,
}
### BASELINE
fixed_no_lsp = {
    's_kl': tune.uniform(0, 0.3), # 0.1
    "s_decay": tune.loguniform(5e-6, 1e-3),
    "s_lr": tune.loguniform(5e-4,1e-1),
    "nbhd_size": 1, # not needed
    'kernel': torch.dot, # not needed
    "s_lsp": 0,
}

best_fixed_no_lsp_cora = {
    's_kl': 0.3, # 0.1
    "s_lr": 5e-2 ,
    "s_decay": 4e-4,
    "nbhd_size": 1, # not needed
    'kernel': torch.dot, # not needed
    "s_lsp": 0,
}

fixed_with_lsp_cora = {
    's_kl': 0.3, # 0.1
    "s_lr": 5e-2 ,
    "s_decay": 4e-4,
    "nbhd_size": 1, # not needed
    'kernel': torch.dot,
    "s_lsp": tune.loguniform(0.1,100),
}

best_fixed_with_lsp_cora = {
    's_kl': 0.3, # 0.1
    "s_lr": 5e-2 ,
    "s_decay": 4e-4,
    "nbhd_size": 1, # not needed
    'kernel': torch.dot,# rbf
    "s_lsp": 80,
}

best_fixed_no_lsp_citeseer = {
    's_kl': 0.18, # 0.1
    "s_lr": 4e-2,
    "s_decay": 1e-4,
    "nbhd_size": 1, # not needed
    'kernel': torch.dot, # not needed
    "s_lsp": 0,
}

fixed_with_lsp_citeseer = {
    's_kl': 0.18, # 0.1
    "s_lr": 4e-2,
    "s_decay": 1e-4,
    "nbhd_size": 1, # not needed
    'kernel': torch.dot, # not needed
    "s_lsp": tune.loguniform(1,100),
}

best_fixed_with_lsp_citeseer = {
    's_kl': 0.18, # 0.1
    "s_lr": 4e-2,
    "s_decay": 1e-4,
    "nbhd_size": 1, # not needed
    'kernel': torch.dot, # not needed
    "s_lsp": 10,
}
best_fixed_no_lsp_pubmed = {
    's_kl': 0.25, # 0.1
    "s_lr": 4e-2 ,
    "s_decay": 1e-4,
    "nbhd_size": 1, # not needed
    'kernel': torch.dot, # not needed
    "s_lsp": 0,
}
fixed_with_lsp_pubmed = {
    's_kl': 0.25, # 0.1
    "s_lr": 4e-2 ,
    "s_decay": 1e-4,
    "nbhd_size": 1, # not needed
    'kernel': torch.dot, # not needed
    "s_lsp": tune.loguniform(1,100),
}
best_fixed_with_lsp_pubmed = {
    's_kl': 0.25, # 0.1
    "s_lr": 4e-2 ,
    "s_decay": 1e-4,
    "nbhd_size": 1, # not needed
    'kernel': torch.dot, # not needed
    "s_lsp": 1.4,
}
teacher_fixed_cfg = {
    'lr': tune.loguniform(1e-5, 1e-1),
    'decay': tune.loguniform(1e-7, 1e-2)
}
#pretrianed_with_lsp['s_lsp'] = tune.choice([0.1, 1, 5, 10, 50, 100])

#pretrianed_no_lsp = search_cfg.copy()
#pretrianed_no_lsp['s_lsp'] = 0

#joint_with_lsp = search_cfg.copy()
#joint_with_lsp['s_lsp'] = tune.choice([0.1, 1, 5, 10, 50, 100])

#joint_no_lsp = search_cfg.copy()
#joint_no_lsp['s_lsp'] = 0

cfgs = {
    "cora-nolsp": {
        "s_lr": 0.03,
        "s_decay": 5e-4,
        "s_kl": 0.1,
        "pt_epochs": 50,
        "pt_lr": 1e-3,
    },
    "citeseer-nolsp": {
        "s_lr": 0.01,
        "s_decay": 1e-4,
        "s_kl": 0.3,
        'pt_epochs': 80,
        "pt_lr": 1e-3,
    },
    "pubmed-nolsp": {
        "s_lr": 0.015,
        "s_decay": 1e-4,
        "s_kl": 0,
        'pt_epochs': 50,
        "pt_lr": 3e-3,
    }
}

