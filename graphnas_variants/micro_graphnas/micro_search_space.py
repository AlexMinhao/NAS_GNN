import torch
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.nn.conv import *
import numpy as np
import scipy.sparse as sp
import networkx as nx

gnn_list = [
    "gat_8",  # GAT with 8 heads
    "gat_6",  # GAT with 6 heads
    "gat_4",  # GAT with 4 heads
    "gat_2",  # GAT with 2 heads
    "gat_1",  # GAT with 1 heads
    "gcn",  # GCN
    "cheb",  # chebnet
    "sage",  # sage
    "arma",
    "sg",  # simplifying gcn
    "linear",  # skip connection
    "zero",  # skip connection
]
act_list = [
    # "sigmoid", "tanh", "relu", "linear",
    #  "softplus", "leaky_relu", "relu6", "elu"
    "sigmoid", "tanh", "relu", "linear", "elu"
]


def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return F.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")


def gnn_map(gnn_name, in_dim, out_dim, concat=False, bias=True) -> Module:
    '''

    :param gnn_name:
    :param in_dim:
    :param out_dim:
    :param concat: for gat, concat multi-head output or not
    :return: GNN model
    '''
    if gnn_name == "gat_8":
        return GATConv(in_dim, out_dim, 8, concat=concat, bias=bias)
    elif gnn_name == "gat_6":
        return GATConv(in_dim, out_dim, 6, concat=concat, bias=bias)
    elif gnn_name == "gat_4":
        return GATConv(in_dim, out_dim, 4, concat=concat, bias=bias)
    elif gnn_name == "gat_2":
        return GATConv(in_dim, out_dim, 2, concat=concat, bias=bias)
    elif gnn_name in ["gat_1", "gat"]:
        return GATConv(in_dim, out_dim, 1, concat=concat, bias=bias)
    elif gnn_name == "gcn":
        return GCNConv(in_dim, out_dim)
    elif gnn_name == "cheb":
        return ChebConv(in_dim, out_dim, K=2, bias=bias)
    elif gnn_name == "sage":
        return SAGEConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "gated":
        return GatedGraphConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "arma":
        return ARMAConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "sg":
        return SGConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "linear":
        return LinearConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "zero":
        return ZeroConv(in_dim, out_dim, bias=bias)
    # elif gnn_name == "Geo"


class LinearConv(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True):
        super(LinearConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels, bias)

    def forward(self, x, edge_index, edge_weight=None):
        return self.linear(x)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class ZeroConv(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True):
        super(ZeroConv, self).__init__()
        self.out_dim = out_channels


    def forward(self, x, edge_index, edge_weight=None):
        return torch.zeros([x.size(0), self.out_dim]).to(x.device)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SearchSpace(object):
    def __init__(self, search_space=None):
        if search_space:
            self.search_space = search_space
        else:
            self.search_space = {}
            self.search_space["act"] = act_list  # activate function
            self.search_space["gnn"] = gnn_list  # gnn type
            self.search_space["self_index"] = [0, 1]  # 0 means history, 1 means current, each layer contains two input
            self.search_space["concat_type"] = ["add", "product", "concat"]  # same as self_index,
            self.search_space['learning_rate'] = [1e-2, 1e-3, 1e-4, 5e-3, 5e-4]
            self.search_space['dropout'] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            self.search_space['weight_decay'] = [0, 1e-3, 1e-4, 1e-5, 5e-5, 5e-4]
            self.search_space['hidden_unit'] = [8, 16, 32, 64, 128, 256, 512]
            self.search_space['num_layers'] = [2, 3, 4, 5, 6, 7, 8]
        pass

    def get_search_space(self):
        return self.search_space

    @staticmethod
    def generate_action_list(cell=4):
        action_list = []
        for i in range(cell):
            action_list += ["self_index", "gnn"]
        action_list += ["act", "concat_type"]
        return action_list


class IncrementSearchSpace(object):
    def __init__(self, search_space=None, max_cell=10):
        if search_space:
            self.search_space = search_space
        else:
            self.search_space = {}
            self.search_space["act"] = act_list  # activate function
            self.search_space["gnn"] = gnn_list  # gnn type
            for i in range(max_cell):
                self.search_space[f"self_index_{i}"] = list(range(2+i))  # 0 means history, 1 means current, each layer contains two input
            self.search_space["concat_type"] = ["add", "product", "concat"]  # same as self_index,
            self.search_space['learning_rate'] = [1e-2, 1e-3, 1e-4, 5e-3, 5e-4]
            self.search_space['dropout'] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            self.search_space['weight_decay'] = [0, 1e-3, 1e-4, 1e-5, 5e-5, 5e-4]
            self.search_space['hidden_unit'] = [8, 16, 32, 64, 128, 256, 512]
            self.search_space['num_layers'] = [2, 3, 4, 5, 6, 7, 8]
        pass

    def get_search_space(self):
        return self.search_space

    @staticmethod
    def generate_action_list(cell=4):
        action_list = []
        for i in range(cell):
            action_list += [f"self_index_{i}", "gnn"]
        action_list += ["act", "concat_type"]
        return action_list


class SearchSpaceZeng(object):
    def __init__(self, args, search_space=None):

        self.search_space = {}
        # self.search_space["act"] = #act_list  # activate function
        # self.search_space["gnn"] =  #gnn_list  # gnn type


        # self.search_space["K0"] = list(range(8, 520, 8))
        # self.search_space["K1"] = list(range(8, 520, 8))
        # self.search_space["K2"] = list(range(8, 520, 8))
        # self.search_space["K3"] = list(range(8, 520, 8))
        # self.search_space["K4"] = list(range(8, 520, 8))
        # self.search_space["K5"] = list(range(8, 520, 8))
        # self.search_space["K6"] = list(range(8, 520, 8))
        # self.search_space["K7"] = list(range(8, 520, 8))
        # self.search_space["K8"] = list(range(8, 520, 8))
        if args.num_granularity == 0:
            self.search_space["K0"] = [1,2, 4,8,16, 32, 64, 128, 256, 512, 1024, 2048, 3703]
            self.search_space["K1"] = [1,2, 4,8,16, 32, 64, 128, 256, 512, 1024, 2048, 3703]
            # self.search_space["K1"] = [1,3327]
            self.search_space["K2"] = [1,2, 4,8,16, 32, 64, 128, 256, 512, 1024, 2048, 3703]
            self.search_space["K3"] = [1,2, 4,8,16, 32, 64, 128, 256, 512, 1024, 2048, 3703]
            self.search_space["K4"] = [1,2, 4,8,16, 32, 64, 128, 256, 512, 1024, 2048, 3703]
            self.search_space["K5"] = [1,2, 4,8,16, 32, 64, 128, 256, 512, 1024, 2048, 3703]
            self.search_space["K6"] = [1,2, 4,8,16, 32, 64, 128, 256, 512, 1024, 2048, 3703]
            self.search_space["K7"] = [1,2, 4,8,16, 32, 64, 128, 256, 512, 1024, 2048, 3703]
            self.search_space["K8"] = [1,2, 4,8,16, 32, 64, 128, 256, 512, 1024, 2048, 3703]
        else:
            self.search_space["K0"] = list(range(args.num_granularity, 520, args.num_granularity))
            self.search_space["K1"] = list(range(args.num_granularity, 520, args.num_granularity))
            # self.search_space["K1"] = [1,3327]
            self.search_space["K2"] = list(range(args.num_granularity, 520, args.num_granularity))
            self.search_space["K3"] = list(range(args.num_granularity, 520, args.num_granularity))
            self.search_space["K4"] = list(range(args.num_granularity, 520, args.num_granularity))
            self.search_space["K5"] = list(range(args.num_granularity, 520, args.num_granularity))
            self.search_space["K6"] = list(range(args.num_granularity, 520, args.num_granularity))
            self.search_space["K7"] = list(range(args.num_granularity, 520, args.num_granularity))
            self.search_space["K8"] = list(range(args.num_granularity, 520, args.num_granularity))
            self.search_space["K0"].append(3703)
            self.search_space["K1"].append(3703)
            self.search_space["K2"].append(3703)
            self.search_space["K3"].append(3703)
            self.search_space["K4"].append(3703)
            self.search_space["K5"].append(3703)
            self.search_space["K6"].append(3703)
            self.search_space["K7"].append(3703)
            self.search_space["K8"].append(3703)
            self.search_space["K0"].append(1)
            self.search_space["K1"].append(1)
            self.search_space["K2"].append(1)
            self.search_space["K3"].append(1)
            self.search_space["K4"].append(1)
            self.search_space["K5"].append(1)
            self.search_space["K6"].append(1)
            self.search_space["K7"].append(1)
            self.search_space["K8"].append(1)
        # self.search_space["K0_act"] = ['linear', 'identity']
        # self.search_space["K1_act"] = ['linear', 'identity']
        # self.search_space["K2_act"] = ['linear', 'identity']
        # self.search_space["K3_act"] = ['linear', 'identity']
        # self.search_space["K4_act"] = ['linear', 'identity']
        # self.search_space["K5_act"] = ['linear', 'identity']
        # self.search_space["K6_act"] = ['linear', 'identity']
        # self.search_space["K7_act"] = ['linear', 'identity']
        # self.search_space["K8_act"] = ['linear', 'identity']

        # self.search_space["K0"] = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152,
        #                            160, 168, 176, 182, 190, 198, 206, 214, 222, 230, 238, 248, 256, 264, 272, 280, 288,
        #                            296, 304, 312, 320, 328, 336, 344, 340, 348, 356, 412, 464, 472, 480, 488, 504, 512]
        # self.search_space["K1"] = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152,
        #                            160, 168, 176, 182, 190, 198, 206, 214, 222, 230, 238, 248, 256, 264, 272, 280, 288,
        #                            296, 304, 312, 320, 328, 336, 344, 340, 348, 356, 412, 464, 472, 480, 488, 504, 512]
        # self.search_space["K2"] = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152,
        #                            160, 168, 176, 182, 190, 198, 206, 214, 222, 230, 238, 248, 256, 264, 272, 280, 288,
        #                            296, 304, 312, 320, 328, 336, 344, 340, 348, 356, 412, 464, 472, 480, 488, 504, 512]
        # self.search_space["K3"] = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152,
        #                            160, 168, 176, 182, 190, 198, 206, 214, 222, 230, 238, 248, 256, 264, 272, 280, 288,
        #                            296, 304, 312, 320, 328, 336, 344, 340, 348, 356, 412, 464, 472, 480, 488, 504, 512]
        # self.search_space["K4"] = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152,
        #                            160, 168, 176, 182, 190, 198, 206, 214, 222, 230, 238, 248, 256, 264, 272, 280, 288,
        #                            296, 304, 312, 320, 328, 336, 344, 340, 348, 356, 412, 464, 472, 480, 488, 504, 512]
        # self.search_space["K5"] = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152,
        #                            160, 168, 176, 182, 190, 198, 206, 214, 222, 230, 238, 248, 256, 264, 272, 280, 288,
        #                            296, 304, 312, 320, 328, 336, 344, 340, 348, 356, 412, 464, 472, 480, 488, 504, 512]
        # self.search_space["K6"] = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152,
        #                            160, 168, 176, 182, 190, 198, 206, 214, 222, 230, 238, 248, 256, 264, 272, 280, 288,
        #                            296, 304, 312, 320, 328, 336, 344, 340, 348, 356, 412, 464, 472, 480, 488, 504, 512]
        # self.search_space["K7"] = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152,
        #                            160, 168, 176, 182, 190, 198, 206, 214, 222, 230, 238, 248, 256, 264, 272, 280, 288,
        #                            296, 304, 312, 320, 328, 336, 344, 340, 348, 356, 412, 464, 472, 480, 488, 504, 512]
        # self.search_space["K8"] = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152,
        #                            160, 168, 176, 182, 190, 198, 206, 214, 222, 230, 238, 248, 256, 264, 272, 280, 288,
        #                            296, 304, 312, 320, 328, 336, 344, 340, 348, 356, 412, 464, 472, 480, 488, 504, 512]
        # self.search_space["self_index"] = [0, 1]  # 0 means history, 1 means current, each layer contains two input
        # self.search_space["concat_type"] = ["add", "product", "concat"]  # same as self_index,
        # self.search_space['learning_rate'] = [1e-2, 1e-3, 1e-4, 5e-3, 5e-4]
        # self.search_space['dropout'] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # self.search_space['weight_decay'] = [0, 1e-3, 1e-4, 1e-5, 5e-5, 5e-4]
        # self.search_space['hidden_unit'] = [8, 16, 32, 64, 128, 256, 512]
        # self.search_space['num_layers'] = [2, 3, 4, 5, 6, 7, 8]
        pass

    def get_search_space(self):
        return self.search_space


    def generate_action_list(self,K = 4):
        action_list = []

        for key in self.search_space:
            if len(action_list) < int(K):
                action_list += [key]

        # action_list += ["act"]
        return action_list









if __name__ == "__main__":
    obj = IncrementSearchSpace()
    print(obj.generate_action_list())
    print(obj.get_search_space())
