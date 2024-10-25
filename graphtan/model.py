import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch_geometric.data import Data
from models.layers import StructuralAttentionLayer, TemporalAttentionLayer, GraphAttentionV2Layer
from node2vec import Node2Vec
from torch_geometric.nn import SAGPooling, GATConv, GCNConv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.stats as stats
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from scipy import spatial
import pandas as pd
import torch_geometric.utils as pyg_utils
from scipy.stats import wasserstein_distance

class GraphTAN(nn.Module):
    def __init__(self, args, num_features, time_length):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(GraphTAN, self).__init__()
        self.args = args
        if args.window < 0:
            self.num_time_steps = time_length
        else:
            self.num_time_steps = min(time_length, args.window + 1)  # window = 0 => only self.
        self.num_features = num_features
        self.final_graph_embedding = nn.Parameter(torch.zeros((self.num_time_steps, args.embedding_dim)))
        self.structural_head_config = list(map(int, args.structural_head_config.split(",")))
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(",")))
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))
        self.spatial_drop = args.spatial_drop
        self.temporal_drop = args.temporal_drop
        self.graph_embedding = args.embedding_dim
        self.structural_attn, self.temporal_attn = self.build_model()
        self.temporal = args.temporal
        self.pooling = args.pooling
        self.SAGPooling1 = SAGPooling(256, ratio=0.5, GNN=GCNConv)
        self.SAGPooling2 = SAGPooling(256, ratio=0.7, GNN=GATConv)
        self.lin1 = torch.nn.Linear(256, 256)
        self.bceloss = BCEWithLogitsLoss()

    def forward(self, graphs, graphs_emb):
        # Structural Attention forward
        structural_out = []
        graph_list = []
        final_out = []
        for t in range(0, len(graphs)):
            output_layer_0 = self.structural_attn.structural_layer_0(graphs[t])
            if self.pooling:
                x, edge_index, _, batch, _, _ = self.SAGPooling1(output_layer_0.x, output_layer_0.edge_index, None, output_layer_0.batch)
                output_layer_0.x = x
                output_layer_0.edge_index = edge_index
                output_layer_0.batch = batch
                output_layer_1 = self.structural_attn.structural_layer_1(output_layer_0).x
                output_layer_1 = self.lin1(output_layer_1)
            else:
                output_layer_1 = self.structural_attn.structural_layer_1(output_layer_0).x
                output_layer_1 = self.lin1(output_layer_1)

            graph_level_representation = torch.mean(output_layer_1, dim=0)

            structural_out.append(graph_level_representation)

        if self.temporal:
            if(len(graphs_emb) == 0):
                for graph in structural_out:
                    transformed_graph = graph.view(1, 1, -1)
                    graph_list.append(transformed_graph)
                graph_tensor = torch.cat(graph_list, dim=1)
            else:
                for graph in graphs_emb:
                    transformed_graph = graph.view(1, 1, -1)
                    graph_list.append(transformed_graph)
                for graph in structural_out:
                    transformed_graph = graph.view(1, 1, -1)
                    graph_list.append(transformed_graph)
                graph_tensor = torch.cat(graph_list, dim=1)

            # # Temporal Attention forward
            now_time_steps = len(graph_list)
            if(now_time_steps == self.num_time_steps):
                graph_list.clear()
                temporal_out = self.temporal_attn(graph_tensor)
                for i in range(temporal_out.shape[1]):
                    feature = temporal_out[0, i, :]
                    final_out.append(feature)

            embeddings_tensor = torch.stack(final_out)
            self.final_graph_embedding.data = embeddings_tensor.detach()
        else:
            embeddings_tensor = torch.stack(structural_out)
            self.final_graph_embedding.data = embeddings_tensor.detach()
            final_out = structural_out
        return final_out

    def build_model(self):
        input_dim = self.graph_embedding

        # 1: Structural Attention Layers
        structural_attention_layers = nn.Sequential()
        for i in range(len(self.structural_layer_config)):
            layer = StructuralAttentionLayer(input_dim=input_dim,
                                             output_dim=self.structural_layer_config[i],
                                             n_heads=self.structural_head_config[i],
                                             attn_drop=self.spatial_drop,
                                             ffd_drop=self.spatial_drop,
                                             residual=self.args.residual)
            structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer)
            input_dim = self.structural_layer_config[i]
        
        # 2: Temporal Attention Layers
        # input_dim = self.structural_layer_config[-1]
        temporal_attention_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(input_dim=self.temporal_layer_config[i],
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.args.residual)
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]

        return structural_attention_layers, temporal_attention_layers

    def get_emb(self, feed_dict, graphs_emb):
        graphs = feed_dict
        final_emb = self.forward(graphs, graphs_emb)
        return final_emb

    def get_loss(self, graph_emb, real_matrix):
        total_loss = 0.0
        num_samples = 0

        predicted_similarity = torch.zeros((len(graph_emb), len(graph_emb))).to("cuda:0")
        for i in range(0, len(graph_emb)):
            for j in range(0, len(graph_emb)):
                if(i == j):
                    predicted_similarity[i][j] = -100
                else:
                    A = graph_emb[i].reshape(1, -1)
                    B = graph_emb[j].reshape(1, -1)
                    similarity = F.cosine_similarity(A, B)
                    predicted_similarity[i][j] = similarity
                    loss = (predicted_similarity[i][j] - real_matrix[i][j]) ** 2
                    total_loss = total_loss + loss
                    num_samples += 1

        similarity_loss = total_loss / num_samples
        embeddings_tensor = torch.stack(graph_emb)
        final_out = embeddings_tensor.cpu().detach().numpy()
        return similarity_loss, final_out