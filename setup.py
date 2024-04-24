from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
import torch
import argparse
import yaml
from models import GCN
import os

def extract_data(dataset, device):
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    if dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    g = data[0]
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
    classes = data.num_classes
    return g, features, labels, masks, classes

def get_teachers(dataset, input_dim, output_dim, device):
    path = os.getcwd() + "/pt-teachers/gcn_" + dataset
    paths = [(path + "_" + str(num) + "/checkpoint.pt") for num in [25,75,100]] # should be 33, 66, but i named wrong, oops

    teachers = [GCN(3, input_dim, 128, output_dim).to(device) for _ in range(4)]
    for i,p in enumerate(paths):
        teachers[i+1].load_state_dict(torch.load(p)['model_state_dict'])

    return teachers



def get_training_params(path, dataset):
    with open(path, 'r') as f:
        loaded_data = yaml.load(f, Loader=yaml.FullLoader)
    return loaded_data[dataset]

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="pubmed",
        help="Dataset name ('cora', 'citeseer', 'pubmed').",
    )

    parser.add_argument(
        "--path",
        type=str,
        default="results",
        help="Location to store, relative to current dir. Defuault: results",
    )

    parser.add_argument(
        "--name",
        type=str,
        help="Name of experiment to run",
    )

    parser.add_argument(
        "--samples",
        type=str,
        default=100,
        help="Number of hyperparam configs to test",
    )

    parser.add_argument(
        "--stopping",
        type=str,
        default=300,
        help="Max number of epochs",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Train or Tune. Defualt: Train",
    )

    parser.add_argument(
        "--params",
        type=str,
        default="train",
        help="Path to hyperparameter YAML file to use during training.",
    )

    return parser
