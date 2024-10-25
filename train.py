import argparse
import networkx as nx
import numpy as np
import torch.nn as nn
import dill
import pickle as pkl
import scipy
import traceback
from torch.utils.data import DataLoader
import torch.autograd as autograd
from torch.autograd import detect_anomaly
import time
from utils.minibatch import  MyDataset
from utils.utilities import to_device
from eval.link_prediction import evaluate_classifier
from models.model import GraphTAN
from visualizations import similarity_heatmap
from visualizations import plot_trend_over_time
from anomaly_evaluation import evaluate_anomalies
import gc
import copy
import torch
import torch.nn.functional as F
import random
from similarity_ranknig_measures import eval_similarity
from datetime import datetime as dt
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from bokeh.plotting import show
import holoviews as hv
torch.autograd.set_detect_anomaly(True)


def inductive_graph(graph_former, graph_later):
    """Create the adj_train so that it includes nodes from (t+1)
       but only edges from t: this is for the purpose of inductive testing.

    Args:
        graph_former ([type]): [description]
        graph_later ([type]): [description]
    """
    newG = nx.MultiGraph()
    newG.add_nodes_from(graph_later.nodes(data=True))
    newG.add_edges_from(graph_former.edges(data=False))
    return newG


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, nargs='?', default=0,
                        help="total time steps used for train, eval and test")
    # Experimental settings.
    parser.add_argument('--dataset', type=str, nargs='?', default='formula',
                        help='dataset name')
    parser.add_argument('--dataset_name', type=str, nargs='?', default='formula_2019_graphs_dynamic',
                        help='dataset name')
    parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                        help='GPU_ID (0/1 etc.)')
    parser.add_argument('--epochs', type=int, nargs='?', default=50,
                        help='# epochs')
    parser.add_argument('--batch_size', type=int, nargs='?', default=5,
                        help='Batch size (# nodes)')
    parser.add_argument('--featureless', type=bool, nargs='?', default=True,
                    help='True if one-hot encoding.')
    parser.add_argument('--residual', type=bool, nargs='?', default=True,
                        help='Use residual')
    parser.add_argument('--pooling', type=bool, nargs='?', default=True,
                        help='Use pooling')
    parser.add_argument('--temporal', type=bool, nargs='?', default=True,
                        help='Use temporal')
    # Weight for negative samples in the binary cross-entropy loss function.
    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.01,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
                        help='Spatial (structural) attention Dropout (1 - keep probability).')
    parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                        help='Temporal attention Dropout (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                        help='Initial learning rate for self-attention model.')
    # Architecture params
    parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,8',
                        help='Encoder layer config: # attention heads in each GAT layer')
    parser.add_argument('--structural_layer_config', type=str, nargs='?', default='256,256',
                        help='Encoder layer config: # units in each GAT layer')
    parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
                        help='Encoder layer config: # attention heads in each Temporal layer')
    parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='256',
                        help='Encoder layer config: # units in each Temporal layer')
    parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                        help='Position wise feedforward')
    parser.add_argument('--embedding_dim', type=int, nargs='?', default='256',
                        help='# graph_embedding_dim')
    parser.add_argument('--structural_head', type=int, nargs='?', default='16',
                        help='Encoder layer config: # attention heads in each GAT layer')
    args = parser.parse_args()
    print(args)

    graphs, adjs, similarity_matrix, graphs_keys, similarity_matrix_gt = load_graphs(args.dataset, args.dataset_name)
    args.time_steps = len(graphs)

    similarity_matrix = similarity_matrix.values
    normalized_matrix = np.zeros_like(similarity_matrix)
    for i in range(similarity_matrix.shape[0]):
        row = similarity_matrix[i, :]
        min_val = np.min([val for idx, val in enumerate(row) if idx != i])
        max_val = np.max([val for idx, val in enumerate(row) if idx != i])
        if min_val == max_val:
            normalized_row = row + 1e-10
        else:
            normalized_row = (row - min_val + 1e-10) / (max_val - min_val)
        normalized_row[i] = -100
        normalized_matrix[i, :] = normalized_row

    if args.featureless == True:
        feats = []
        for x in adjs:
            features = scipy.sparse.lil_matrix((x.shape[0], args.embedding_dim), dtype=np.float32)
            random_values = np.random.rand(x.shape[0], args.embedding_dim)
            identity_matrix = scipy.sparse.csr_matrix(random_values)
            feats.append(identity_matrix)

    assert args.time_steps <= len(adjs), "Time steps is illegal"

    # build dataloader and model and adamw
    device = torch.device("cuda:0")
    dataset = MyDataset(args, graphs, feats, adjs)

    max_num_nodes = max([graph.number_of_nodes() for graph in graphs])

    expanded_out = torch.zeros(max_num_nodes, 256)
    for t in range(len(dataset.pyg_graphs)):
        dataset.pyg_graphs[t].batch = torch.zeros(dataset.pyg_graphs[t].x.shape[0], dtype=torch.long)

    model = GraphTAN(args, max_num_nodes, args.time_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # train
    best_five = 0
    best_patten = 0
    best_pattwenty = 0
    best_kendalltau = 0
    best_spearman = 0
    best_mrr = 0
    best_map10 = 0
    patient = 0

    for epoch in range(args.epochs):
        start = time.time()
        dataloader = DataLoader(dataset.pyg_graphs,
                                batch_size=args.batch_size)
        model.train()
        all_time = 0
        epoch_loss = []
        feed_dict = []
        now_graph = 0
        now_emb = []
        graphs_emb = []
        predicted_similarity = []
        detached_graphs_emb = []
        real_matrix = torch.tensor(
            normalized_matrix, dtype=torch.float32, requires_grad=True)

        for batch in dataloader.dataset:
            graph = copy.deepcopy(batch)
            feed_dict.append(graph)
            now_graph += 1
            if(now_graph % args.time_steps == 0 or batch == (len(dataloader.dataset) - 1)):
                opt.zero_grad()
                feed_dict = to_device(feed_dict, device)
                now_emb = model.get_emb(feed_dict, graphs_emb)
                graphs_emb.clear()
                for i in range(len(now_emb)):
                    graphs_emb.append(now_emb[i])
                loss, final_out = model.get_loss(graphs_emb, real_matrix, google_trends_df)
                loss.backward()
                opt.step()

        epoch_loss.append(loss.item())
        model.eval()
        eval = eval_similarity(graph_embs=final_out, times=graphs_keys,
                                      similarity_matrix_gt=similarity_matrix_gt)

        print("Epoch {:<3},  Loss = {:.6f}".format(epoch, np.mean(epoch_loss)))
        print("p@5 {:.6f} p@10 {:.6f} mrr {:.6f} map@10 {:.6f}".format(eval.values[0][1], eval.values[1][1],
                                                                       eval.values[4][1], eval.values[3][1]))
        # save the best model
        if(eval.values[4][1] > best_mrr):
            best_five = eval.values[0][1]
            best_patten = eval.values[1][1]
            best_mrr = eval.values[4][1]
            best_map10 = eval.values[3][1]
            best_spearman = spearman
            torch.save({
                'state_dict': model.state_dict(),
            }, f"./model_checkpoints/{args.dataset}/{args.dataset_name}.pt")
            print("------save best model successfully------")

        print("best_p@5 {:.6f} best_p@10 {:.6f} best_mrr {:.6f} best_map@10 {:.6f} best_Spearman {:.6f}".format(best_five, best_patten, best_mrr, best_map10, best_spearman))
        print("use_time: {:<3}".format(time.time() - start))