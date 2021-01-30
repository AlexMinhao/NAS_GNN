import torch
import torch.nn as nn
import torch.nn.functional as F

from graphnas_variants.micro_graphnas.micro_search_space import gnn_map, act_map
from graphnas_variants.micro_graphnas.utils import low_dim_all


class MicroGNN(nn.Module):
    def __init__(self, action, num_feat, num_classes, num_hidden, dropout=0.6, layers=3, stem_multiplier=2, bias=True):
        super(MicroGNN, self).__init__()
        self._layers = layers
        self.dropout = dropout                   # 3703      32        3703       32
        his_dim, cur_dim, hidden_dim, out_dim = num_feat, num_feat, num_hidden, num_hidden
        self.cells = nn.ModuleList()
        for i in range(layers): # 2
            cell = Cell(action, his_dim, cur_dim, hidden_dim, out_dim, concat=False, bias=bias)
            self.cells += [cell]
            his_dim = cur_dim
            cur_dim = cell.multiplier * out_dim if action[-1] == "concat" else out_dim

        self.classifier = nn.Linear(cur_dim, num_classes)
        pass

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        s0 = s1 = x # torch.Size([3327, 3703])
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, edge_index, self.dropout)
        out = s1
        logits = self.classifier(out.view(out.size(0), -1))
        return logits



class ZengGNN(nn.Module):
    def __init__(self, action, num_feat, num_classes, adj, dropout=0.6, layers=3):
        super(ZengGNN, self).__init__()
        self._layers = layers
        self.dropout = dropout                   # 3703      32        3703       32
        self.nclass = num_classes
        self.actions = action
        self.cells = nn.ModuleList()
        self.all_low_dim_channel = 0
        self.adj = adj

        self.adj_numpy = adj.cpu().to_dense().numpy()
        self.adj_dist = []
        # self.K = action[:-1]
        # self.act = action[-1:]
        adj_d = low_dim_all(self.adj_numpy, len(action), eye=False)
        for k in range(len(self.actions)):
            # adj_d = low_dim_all(self.adj_numpy, k + 1, eye=False)
            self.adj_dist.append(adj_d[k])
        for c in self.actions:
            self.all_low_dim_channel += c

        for i in range(layers):  # 2
            cell = ZengCell(action, num_feat, self.adj_dist)
            self.cells += [cell]

        self.classifier = nn.Linear(self.all_low_dim_channel, self.nclass)  # bias?


    def forward(self, x, adj):

        # x = F.dropout(x, p=self.dropout, training=self.training)
        s0 = x  # torch.Size([3327, 3703])
        for i, cell in enumerate(self.cells):
            s0 = cell(s0, adj)
        out = s0
        logits = self.classifier(out.view(out.size(0), -1))
        return logits



class ZengCell(nn.Module):

    def __init__(self, action_list, nfeat, adj):
        '''

        :param action_list: like ['self_index', 'gnn'] * n +['act', 'concat_type']
        :param his_dim:
        :param cur_dim:
        :param hidden_dim:
        :param out_dim:
        :param concat:
        :param bias:
        '''

        super(ZengCell, self).__init__()
        # print(his_dim, cur_dim, hidden_dim, out_dim)
        self.nfeat = nfeat

        self._gnn = nn.ModuleList()
        self._compile(action_list)
        self.adj_dist = adj

    def _compile(self, action_list):
        self.K_hops = action_list# [0, 'sage', 0, 'gat_6']

        # for i, action in enumerate(cells_info):
        #     if i % 2 == 0:
        #         self._indices.append(action)  # action is a indice
        #     else:
        #         self._gnn.append(gnn_map(action, self.hidden_dim, self.out_dim, self.concat_of_multihead, self.bias))
        self.conv_k = nn.ModuleList()
        for c in self.K_hops:
               self.conv_k.append(nn.Linear(self.nfeat, c))



    def forward(self, x, adj):

        all_features = []
        # adj_numpy = adj.cpu().to_dense().numpy()
        for k in range(len(self.K_hops)):
            # adj_dist = low_dim_all(adj_numpy, k+1, eye = False)

            low_feat = torch.spmm(self.adj_dist[k], x)
            # Check the number of neighbors
            all_features.append(low_feat)
        # print("adj * feature = done !")
        state = []
        for i, c in enumerate(self.K_hops):

            if c == 3327:
                s = all_features[i]
            else:
                s = self.conv_k[i](all_features[i])

            # low_feature.append(self.activate(low_feat))
            state.append(s)

        out = torch.cat(state, dim=1)
        # print("dimension  = done !")
        return out


class Cell(nn.Module):

    def __init__(self, action_list, his_dim, cur_dim, hidden_dim, out_dim, concat, bias=True):
        '''

        :param action_list: like ['self_index', 'gnn'] * n +['act', 'concat_type']
        :param his_dim:
        :param cur_dim:
        :param hidden_dim:
        :param out_dim:
        :param concat:
        :param bias:
        '''
        assert hidden_dim == out_dim  # current version only support this situation
        super(Cell, self).__init__()
        self.his_dim = his_dim  #  3703
        self.cur_dim = cur_dim  #  3703
        self.hidden_dim = hidden_dim  # 32
        self.out_dim = out_dim  # 32
        self.concat_of_multihead = concat  # False
        self.bias = bias
        # print(his_dim, cur_dim, hidden_dim, out_dim)

        self.preprocess0 = nn.Linear(his_dim, hidden_dim, bias)
        self.preprocess1 = nn.Linear(cur_dim, hidden_dim, bias)

        self._indices = []
        self._gnn = nn.ModuleList()
        self._compile(action_list)

    # def _compile(self, action_list):
    #     cells_info = action_list[:-2]
    #     assert len(cells_info) % 2 == 0
    #
    #     self._steps = len(cells_info) // 2
    #     self.multiplier = self._steps //2
    #     self._act = act_map(action_list[-2])
    #     self._concat = action_list[-1]
    #
    #     for i, action in enumerate(cells_info):
    #         if i % 2 == 0:
    #             self._indices.append(action)  # action is a indice
    #         else:
    #             self._gnn.append(gnn_map(action, self.hidden_dim, self.out_dim, self.concat_of_multihead, self.bias))
    # Version 1, every conv_opt take two input (contain two GNN)
    # def forward(self, s0, s1, edge_index, drop_prob):
    #     s0 = self.preprocess0(s0)
    #     s1 = self.preprocess1(s1)
    #
    #     states = [s0, s1]
    #     for i in range(self.multiplier):
    #         h1 = states[self._indices[2 * i]]
    #         h2 = states[self._indices[2 * i + 1]]
    #         op1 = self._gnn[2 * i]
    #         op2 = self._gnn[2 * i + 1]
    #         h1 = op1(h1, edge_index)
    #         h2 = op2(h2, edge_index)
    #         s = h1 + h2
    #         s = F.dropout(s, p=drop_prob, training=self.training)
    #         states += [s]
    #     return self._act(torch.cat(states[2:], dim=1))

    # Version 2, every conv_opt take one input (contain one GNN)
    def _compile(self, action_list):
        cells_info = action_list[:-2] # [0, 'sage', 0, 'gat_6']
        assert len(cells_info) % 2 == 0

        self._steps = len(cells_info) // 2
        self.multiplier = self._steps
        self._act = act_map(action_list[-2])
        self._concat = action_list[-1]

        for i, action in enumerate(cells_info):
            if i % 2 == 0:
                self._indices.append(action)  # action is a indice
            else:
                self._gnn.append(gnn_map(action, self.hidden_dim, self.out_dim, self.concat_of_multihead, self.bias))

    def forward(self, s0, s1, edge_index, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[i]]
            op1 = self._gnn[i]
            s = op1(h1, edge_index) # s3
            # s = F.dropout(s, p=drop_prob, training=self.training)
            states += [s]
        if self._concat == "concat":
            return self._act(torch.cat(states[2:], dim=1))
        else:
            tmp = states[2]
            for i in range(2,len(states)):
                if self._concat == "add":
                    tmp = torch.add(tmp, states[i])
                elif self._concat == "product":
                    tmp = torch.mul(tmp, states[i])
            return tmp

