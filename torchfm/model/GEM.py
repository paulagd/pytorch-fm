import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchfm.layer import GraphConvolution, multiply_context_gcn, compute_pairwise
from IPython import embed


class GCN(nn.Module):
    def __init__(self, nfeat, embedding_size, dropout, active_layers, initial_W, nhid=128):
        super(GCN, self).__init__()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if int(active_layers) > 1:
            self.gc1 = GraphConvolution(nfeat, embedding_size, initial_W)
            self.gc2 = GraphConvolution(nfeat, embedding_size, initial_W)
            self.active2 = True
            # self.dropout2 = nn.Dropout(dropout)
        else:
            self.gc1 = GraphConvolution(nfeat, embedding_size, initial_W)
            self.active2 = False

    def forward(self, features, adj):

        # features = [5032 x 5032]
        # adj (normalized matrix) = [5032 x 5032]
        # IDEA: Remove nonlinear functions
        # embeddings = F.relu(self.gc1(features, adj))
        embeddings = self.gc1(features, adj)
        # [5032 x 64]
        embeddings = self.dropout1(embeddings)
        #  [5032 x 64]
        if self.active2:
            embeddings = self.gc2(embeddings, adj)
            #TODO: HACER RELU Y DO EN LA SEGUNDA CAPA TAMBIEN
            # embeddings = self.gc2(embeddings, adj)
            # # [5032 x 64]
            # embeddings = self.dropout2(embeddings)

        return embeddings


class BPR_GEM(nn.Module):

    def __init__(self, embedding_size, features, norm_train_mat, dropout=0.5, active_layers=1, context=False,
                 pairwise=False, initial_W=None):
        super(BPR_GEM, self).__init__()
        self.features = features
        self.context = context
        self.pairwise = pairwise
        self.norm_train_mat = norm_train_mat

        self.GCN = GCN(features.shape[1], embedding_size, dropout, active_layers, initial_W)

    def forward(self, u, i, j, c):

        embeddings = self.GCN(self.features, self.norm_train_mat)

        user = embeddings[u]
        item_i = embeddings[i]
        item_j = embeddings[j]

        if self.context:
            if self.pairwise:
                prediction_i = compute_pairwise(user, item_i, embeddings, c, gcn_flag=True).sum(dim=-1)
                prediction_j = compute_pairwise(user, item_j, embeddings, c, gcn_flag=True).sum(dim=-1)
            else:
                ui = user * item_i
                uj = user * item_j
                prediction_i = multiply_context_gcn(ui, embeddings, c).sum(dim=-1)
                prediction_j = multiply_context_gcn(uj, embeddings, c).sum(dim=-1)

        else:
            prediction_i = (user * item_i).sum(dim=-1)
            prediction_j = (user * item_j).sum(dim=-1)

        # return embeddings to draw them optionally with tensorboard
        return prediction_i, prediction_j, embeddings