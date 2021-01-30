import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)

class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.W(input)
        output = torch.spmm(adj, support)
        return output

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.elu = torch.nn.ELU(inplace=True)
    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            #x = F.relu(x)
            x = self.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class Ladder(nn.Module):
    """
    A Simple PyTorch Implementation of One-layer Ladder-Aggregation.
    Assuming the features have been preprocessed with K-step graph propagation (finish spilting process, get x).
    """
    def __init__(self, nfeat, nclass, decay, L, K, dropout):
        super(Ladder, self).__init__()
        self.L = L
        self.K = K
        self.conv_k = nn.ModuleList()
        self.all_low_dim_channel = 0
        for i in range(0, K-L):
            self.low_dim_channel = int(nfeat * decay ** (i+1)) #citeseer:1/5;
            # self.low_dim_channel = nfeat
            if self.low_dim_channel < 1:
                self.low_dim_channel = 1
            print('low dim for each level |nfeat|decay|each dim:',nfeat, decay, self.low_dim_channel)
            self.conv_k.append(nn.Linear(nfeat, self.low_dim_channel))
            self.all_low_dim_channel += self.low_dim_channel
        print('all dim:',self.all_low_dim_channel)
        self.W = nn.Linear(nfeat + self.all_low_dim_channel, nclass) #bias?
        # self.activate = nn.Softmax(-1)
        # self.activate = torch.nn.ELU(inplace=True)
    def forward(self, x):
        low_feature = []
        low_feature.append(x[0])
        for i in range(0, self.K-self.L):
            low_feat = self.conv_k[i](x[i+1])
            # low_feature.append(self.activate(low_feat))
            low_feature.append(low_feat)
        # concat for aggregation
        out = torch.cat(low_feature, dim = 1)
        out = self.W(out)
        return out

def get_model(model_opt, nfeat, nclass, nhid=0, decay=1, L=None, K=None, dropout=0, cuda=True):
    if model_opt == "GCN":
        model = GCN(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=dropout)
    elif model_opt == "SGC":
        model = SGC(nfeat=nfeat,
                    nclass=nclass)
    elif model_opt == "Low":
        model = Ladder(nfeat=nfeat, nclass=nclass, decay=decay, L=L, K=K, dropout=dropout)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if cuda: model.cuda()
    return model
