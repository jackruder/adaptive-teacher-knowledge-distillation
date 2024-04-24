import dgl
import dgl.nn as dglnn
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from dgl import AddSelfLoop

from utils import *

import torch 
import torch.nn as nn

def collect_neighborhood_from_node(graph: dgl.DGLGraph, node, eps):
    """
    input: graph, given node in graph, and eps number of hops
    output: list{set} where each element i of the list gives the nodes i hops away
    """

    nbhd = [{node}]
    visited = set()
    for i in range(1,eps+1):
        nbhd.append(set())
        opn = nbhd[i-1].copy()
        while len(opn) > 0: ## while all nodes not visited
            root = opn.pop()
            visited.add(root)
            s = graph.in_edges(root)[0].tolist()
            p = graph.out_edges(root)[1].tolist()
            for n in s:
                if n not in visited:
                    nbhd[i].add(n)
            for n in p:
                if n not in visited:
                    nbhd[i].add(n)
    return nbhd

def collect_pooled_neighborhood(graph: dgl.DGLGraph, eps, mask):
    """
    input: graph, eps number of hops, node mask.
    output: dict[set], nbhd_mask

    where each d[node] of the matrix is the set of nodes at most eps hops away from the given node
    only nodes specified by the mask are included (including the neighbors. nodes without neighbors are excluded, and a mask with the included neighbors is returned
    """

    d = {}
    for node in graph.nodes()[mask].tolist():
        d[node] = set()
        nbhd = collect_neighborhood_from_node(graph, node, eps)
        for j,nbhd_j in enumerate(nbhd):
            if j > 0:
                for neighbor in nbhd_j:
                    if mask[neighbor]: # if the mask indicates the neighbor
                        d[node].add(neighbor)
    # prune nodes with empty neighborhoods
    d_prune = {}
    mask = [False for _ in range(len(graph.nodes()))]
    for node in d:
        if len(d[node]) > 0:
            d_prune[node] = d[node]
            mask[node] = True 
    return d_prune, mask


def collect_unpooled_neighborhood(graph: dgl.DGLGraph, nodes, eps):
    """
    input: graph, set of nodes, and eps number of hops
    output: dict[list[set]]

    where each d[node][j] of the matrix is the set of nodes exactly j hops away from the given node
    """

    d = {}
    for node in nodes:
        d[node] = collect_neighborhood_from_node(graph, node, eps)
    return d

def local_structure(graph, mask, nbhd_mask, nbhds, layer_features, kernel):
    """ 
    input: 
        graph: graph structure
        mask: mask indicating the subset of nodes considered in training
        nbhd_mask: mask indicating nodes with a nbhd
        nbhd: dict of local neighborhoods to consider
        layer_features: list[nodes] = feature vector,
        eps: size of neighborhood assuming edge weight 1
        kernel: dist func. Can be RBF, Linear, etc.,
    """

    # get neighborhood of node as a set
    # search all nodes eps away,
    nodes = graph.nodes()[nbhd_mask]
    m_pos = translate_node(mask)

    
    S = {} # store each S_i
    dist = {} # store i,j pairs to avoid recomputing
    norms = {}
    for node_i in nodes.tolist(): ## compute kernel function
        z_i = layer_features[m_pos[node_i]]
        norms[node_i] = 0
        for node_j in nbhds[node_i]:
            if (node_j, node_i) not in dist:
                z_j = layer_features[m_pos[node_j]]
                dist[(node_i,node_j)] = torch.exp(kernel(z_i,z_j))
                norms[node_i] += dist[(node_i, node_j)]
            else:
                norms[node_i] += dist[(node_j, node_i)]


    for i, node_i in enumerate(nodes.tolist()): ## compute local structure of node i
        s_i = torch.ones(len(nbhds[node_i]), 1) # initialize structure vector
        for j, node_j in enumerate(nbhds[node_i]):
            if (node_i, node_j) in dist:
                s_i[j] = dist[(node_i, node_j)] / norms[node_i]
            else:
                s_i[j] = dist[(node_j, node_i)] / norms[node_i]
        S[i] = s_i

    return S



def compute_lsp(g, mask, nbhd, mid_features_t, mid_features_s, kernel):
    nbhd_d, nbhd_mask = nbhd
    lsp = 0
    for l in range(len(mid_features_t)): ## for every layer
        lt = local_structure(g, mask, nbhd_mask, nbhd_d, mid_features_t[l], kernel)
        ls = local_structure(g, mask, nbhd_mask, nbhd_d, mid_features_s[l], kernel)
        kldivs = torch.ones(len(lt),1)
        for i in range(len(lt)): # each node i
            # compute mean KL divergence for each node i
            kldivs[i] = torch.sum(torch.mul(ls[i], torch.log(torch.div(ls[i], lt[i]))), dim=0)
        lsp += torch.mean(kldivs)
    return lsp / len(mid_features_t)

if __name__ == '__main__':
    data = CoraGraphDataset(transform=AddSelfLoop())
    g = data[0]
    tmask = g.ndata["train_mask"]
    print(collect_pooled_neighborhood(g, 3, tmask))
