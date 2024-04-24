import os
from threading import active_count
import pandas as pd
from pandas.io.xml import get_data_from_filepath

import torch
import torch.nn as nn
import torch.nn.functional as F

from ray import get, train, tune
from ray.train import Checkpoint
from functools import partial
import tempfile

from structure import collect_pooled_neighborhood, local_structure, compute_lsp
from models import GCN, MLP, evaluate
from utils import *
from setup import *
from rayconfig import *
from gridsearch import single_model_search

CHECKPOINT_FREQ = 10

def train_model_tune(g, architecture, features, labels, masks, classes, params):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.NLLLoss()

    input_dim = features.shape[1]
    output_dim = classes

    model = architecture(3, input_dim, 128, output_dim).to(device)

    lr = params["lr"]
    decay = params["decay"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start_epoch = checkpoint_dict["epoch"] + 1
            model.load_state_dict(checkpoint_dict["model_state_dict"])
            optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

    # training loop
    epoch = 0
    while True:
        model.train()
        logits, _ = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)

        val_loss = loss_fcn(logits[val_mask], labels[val_mask])
        test = evaluate(g, features, labels, masks[2], model)

        metrics = {"val_loss": val_loss.item(), "acc": acc, "epoch": epoch, "test": test}

        if epoch % CHECKPOINT_FREQ == 0:
            with tempfile.TemporaryDirectory() as tempdir:
                checkpoint_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(
                        checkpoint_data,
                        os.path.join(tempdir, "checkpoint.pt"),
                )
                train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))
        else:
            train.report(metrics=metrics)
        epoch += 1


if __name__ == "__main__":
    os.environ["RAY_AIR_NEW_PERSISTENCE_MODE"]="0"

    parser = build_parser()

    parser.add_argument(
        "--model",
        type=str,
        default="GCN",
        help="GCN or MLP",
    )

    args = parser.parse_args()
    # load and preprocess dataset

    if args.model == "GCN":
        arch = GCN
    elif args.model == "MLP":
        arch = MLP
    else:
        raise ValueError("Unknown model architecture: {}".format(args.model))

    storage_path=os.getcwd() + "/" + args.path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g, features, labels, masks, classes = extract_data(args.dataset, device)
    run = partial(train_model_tune, g, arch, features, labels, masks, classes)
    #run = tune.with_resources(run, {"cpu": 1})
    if args.mode == "tune":
        tuner = setup_tuning(run, single_model_search, storage_path, args)
        results = tuner.fit()
        best_result = results.get_best_result("val_loss", "min")
        df = results.get_dataframe()
        df = df[["config/lr", "config/decay", "val_loss", "acc"]]
        df = df[df["acc"] > 0.5].sort_values(by=['acc'], ascending=[False])
        print(df.head(15))
    elif args.mode == "train":
        CHECKPOINT_FREQ = 1
        hparams = get_training_params(os.getcwd() + "/" + args.params, args.dataset)
        trainer = setup_tuning(run, hparams, storage_path, args)
        results = trainer.fit()
        df = results.get_dataframe()
        print(df)
        print("Mean: {:.2f}, SD: {:.2f}".format(df["test"].mean(), df["test"].std()))



