import argparse
import os
from threading import active_count

import dgl
import dgl.nn as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ray import train, tune
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from ray.train import Checkpoint
from functools import partial
import tempfile
import copy

from structure import collect_pooled_neighborhood, local_structure, compute_lsp
from models import GCN, MLP, evaluate
from utils import *
from setup import *
from rayconfig import *
from gridsearch import atkd_lsp_cfg, atkd_no_lsp_cfg, glnn_cfg, lsp_fixed_cfg

CHECKPOINT_FREQ = 25

def train_structured_iter(g, features, labels, masks, teacher, student, nbhds, params, optimizer_t, optimizer_s, fixed_teacher):
    """
    Perform a model update and report the loss
    """
    train_mask = masks[0]
    val_mask = masks[1]

    loss_fcn = nn.NLLLoss()
    div_func = nn.KLDivLoss(reduction="batchmean", log_target=True)

    s_kl = params['s_kl']
    s_lsp = params['s_lsp']
    
    if params['kernel'] == "l2":
        kernel = l2
    elif params['kernel'] == "rbf":
        kernel = partial(rbf, sigma=params['sigma'])
    else:
        kernel = torch.dot

    teacher.train()
    student.train()

    # train teacher on loss and KL divergence
    logits_t, mid_features_t = teacher(g, features, True)
    logits_s, mid_features_s = student(g, features, True)

    #mid_features_s = [feat.detach() for feat in mid_features_s]
    mid_features_t = [feat.detach() for feat in mid_features_t]


    #Compute LSP for student loss
    loss_lsp = torch.tensor(0.0)
    if s_lsp > 0:
        loss_lsp = compute_lsp(g, train_mask, nbhds['train'], mid_features_t, mid_features_s, kernel)

    ## update teacher
    loss_t = torch.tensor(0.0)
    if not fixed_teacher:
        t_kl = params['t_kl']
        logits_s_cpy = logits_s.clone().detach() # use detached clone of student logits
        loss_ce = loss_fcn(logits_t[train_mask], labels[train_mask]) #CE loss
        loss_kl = div_func(logits_s_cpy[train_mask], logits_t[train_mask])
        loss_t = t_kl * loss_ce + (1 - t_kl) * loss_kl##
        optimizer_t.zero_grad()
        loss_t.backward()
        optimizer_t.step()

    #train student on CE, soft targets and LSP loss
    logits_t = logits_t.detach()
    loss_ce = loss_fcn(logits_s[train_mask], labels[train_mask])
    loss_kl = div_func(logits_s[train_mask], logits_t[train_mask])
    loss_s = s_kl * loss_ce + (1-s_kl) * loss_kl + s_lsp*loss_lsp
    optimizer_s.zero_grad()
    loss_s.backward()
    optimizer_s.step()
    
     #compute validation loss
    val_loss_ce = loss_fcn(logits_s[val_mask], labels[val_mask])
    """
    val_loss_kl = div_func(logits_s[val_mask], logits_t[val_mask])
    val_loss_lsp = torch.tensor(0.0)
    if s_lsp > 0:
        val_loss_lsp = (g, val_mask, nbhds['val'], mid_features_t, mid_features_s, kernel)
    val_loss = s_kl * val_loss_ce + (1-s_kl) * val_loss_kl + s_lsp*val_loss_lsp
    """

    return (loss_s.item(), loss_t.item(), loss_kl.item(), loss_lsp.item(), val_loss_ce.item())

def train_structured(teachers, g, features, labels, masks, classes, nbhds, fixed_teacher, params):
    '''
    train teacher to minimize t_KD * L_CE + (1-t_KD) L_KD
    train student to minimize (s_KD) * L_SP + (1-s_KD) L_KD
    '''

    train_mask = masks[0]
    val_mask = masks[1]
    test_mask = masks[2]

    # create GCN model
    input_dim = features.shape[1]
    output_dim = classes

    teacher = copy.deepcopy(teachers[params['pt_ratio']]) ## copy to preserve original teacher
    student = MLP(3, input_dim, 128, output_dim).to(device)

    #do pretraining

    s_lr = params['s_lr']
    s_decay = params['s_decay']
    
    optimizer_t = None
    if not fixed_teacher:
        t_lr = params['t_lr']
        t_decay = params['t_decay']
        optimizer_t = torch.optim.Adam(teacher.parameters(), lr=t_lr, weight_decay=t_decay)

    optimizer_s = torch.optim.Adam(student.parameters(), lr=s_lr, weight_decay=s_decay)

    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start_epoch = checkpoint_dict["epoch"] + 1
            teacher.load_state_dict(checkpoint_dict["teacher_state_dict"])
            student.load_state_dict(checkpoint_dict["student_state_dict"])
            if not fixed_teacher:
                optimizer_t.load_state_dict(checkpoint_dict["optimizer_t_state_dict"])
            optimizer_s.load_state_dict(checkpoint_dict["optimizer_s_state_dict"])


    epoch = 0
    #while True:
    while True:

        loss_s, loss_t, loss_kl, loss_lsp, val_loss = train_structured_iter(g, features, labels, masks, teacher, student, nbhds, params, optimizer_t, optimizer_s, fixed_teacher)
        accT = evaluate(g, features, labels, val_mask, teacher)
        acc = evaluate(g, features, labels, val_mask, student)
        test = evaluate(g, features, labels, test_mask, student)

        metrics = {"val_loss": val_loss, "acc": acc, "accT": accT, "test": test, "epoch": epoch, "loss_lsp": loss_lsp, "loss_kl": loss_kl}

        if epoch % CHECKPOINT_FREQ == 0:
            with tempfile.TemporaryDirectory() as tempdir:
                if not fixed_teacher:
                    checkpoint_data = {
                        "epoch": epoch,
                        "teacher_state_dict": teacher.state_dict(),
                        "student_state_dict": student.state_dict(),
                        "optimizer_t_state_dict": optimizer_t.state_dict(),
                        "optimizer_s_state_dict": optimizer_s.state_dict(),
                    }
                else:
                    checkpoint_data = {
                        "epoch": epoch,
                        "teacher_state_dict": teacher.state_dict(),
                        "student_state_dict": student.state_dict(),
                        "optimizer_s_state_dict": optimizer_s.state_dict(),
                    }
                torch.save(
                        checkpoint_data,
                        os.path.join(tempdir, "checkpoint.pt"),
                )
                train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))
        else:
            train.report(metrics=metrics)

## end training

if __name__ == "__main__":
    os.environ["RAY_AIR_NEW_PERSISTENCE_MODE"]="0"
    parser = build_parser()

    parser.add_argument(
        "--lsp",
        type=str,
        default="True",
        help="Use LSP.",
    )

    parser.add_argument(
        "--fixed",
        type=str,
        default="False",
        help="Keep teacher fixed (no re-training).",
    )

    args = parser.parse_args()
    # load and preprocess dataset

    storage_path=os.getcwd() + "/" + args.path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g, features, labels, masks, classes = extract_data(args.dataset, device)
    nbhds = {
        'train': collect_pooled_neighborhood(g, 1, masks[0]),
        #'val': collect_pooled_neighborhood(g, 1, masks[1]),
    }

    lsp = args.lsp == "yes"
    fixed = args.fixed == "yes"

    teachers = get_teachers(args.dataset, features.shape[1], classes, device)

    run = partial(train_structured, teachers, g, features, labels, masks, classes, nbhds, fixed)

    if args.mode == "tune":

        if lsp and fixed:
            search = lsp_fixed_cfg
        elif (not lsp) and fixed:
            search = glnn_cfg
        elif lsp and (not fixed):
            search = atkd_lsp_cfg
        else:
            search = atkd_no_lsp_cfg


        tuner = setup_tuning(run, search, storage_path, args)
        results = tuner.fit()
        best_result = results.get_best_result("val_loss", "min")
        df = results.get_dataframe()
        df = df[df["acc"] > 0.5].sort_values(by=['acc'], ascending=[False])
        print(df.head(10))

    elif args.mode == "train":
        hparams = get_param_dict(args.params)
        CHECKPOINT_FREQ = 1
        hparams = get_training_params(os.getcwd() + "/" + args.params, args.dataset)
        trainer = setup_tuning(run, hparams, storage_path, args)
        results = trainer.fit()
        df = results.get_dataframe()
        print(df)
        print("Mean: {:.2f}, SD: {:.2f}".format(df["test"].mean(), df["test"].std()))

