import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
class MLP(nn.Module):
    """
    Basic MLP
    """
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio=0.3,
            norm_type="none",
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.sm = nn.LogSoftmax(dim=1)

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))

            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, _g, features, save_mid=False):
        h = features
        mid_features = []
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)

            if l < len(self.layers) - 1:
                mid_features.append(h)
        out = self.sm(h)

        if save_mid:
            return out, mid_features
        else:
            return out, None

class GCN(nn.Module):
    def __init__(self,
                 num_layers,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 dropout_ratio=0.3
                ):
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.sm = nn.LogSoftmax(dim=1)
        # two-layer GCN

        if num_layers == 1:
            self.layers.append(
                dglnn.GraphConv(input_dim, output_dim, activation=F.relu)
            )
        else:
            self.layers.append(
                dglnn.GraphConv(input_dim, hidden_dim, activation=F.relu)
            )
            for _ in range(num_layers - 2):
                self.layers.append(
                    dglnn.GraphConv(hidden_dim, hidden_dim, activation=F.relu)
                )

            self.layers.append(dglnn.GraphConv(hidden_dim, output_dim))
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, g, features, save_mid=False):
        mid_features = []
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            if i < len(self.layers) - 1:
                mid_features.append(h)
        out = self.sm(h) # apply softmax
        if save_mid:
            return out, mid_features
        else:
            return out, None


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits, _ = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
