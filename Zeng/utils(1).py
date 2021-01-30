import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter
import sklearn

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN", gamma=1):
    adj_normalizer = fetch_normalization(normalization)
    if 'Aug' in normalization:
        adj = adj_normalizer(adj, gamma=gamma)
    else:
        adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def low_dim(adj, L, K): #
    adj_K = np.linalg.matrix_power(adj, K)
    # adj_K1 = np.array(adj_K!=0, dtype=np.float32)
    adj_L = np.linalg.matrix_power(adj, L)
    print('zzzz',adj_L)
    adj_L1 = np.array(adj_L!=0, dtype=np.float32)
    adj_base = adj_L1
    adj_d = []
    for i in range(L, K):
        adj_k = np.linalg.matrix_power(adj, i+1)
        adj_k1 = np.array(adj_k!=0, dtype=np.float32)
        adj_diff = adj_k1 - adj_base
        # adj_diff = sp.csr_matrix(adj_diff)
        # adj_normalizer = fetch_normalization('NormAdj')
        # # adj_diff = adj_normalizer(adj_diff, gamma=gamma)
        # adj_diff = adj_normalizer(adj_diff)
        adj_dist = adj_diff*adj_K
        sum1 = np.sum(adj_dist, 1)
        adj_dist = sp.csr_matrix(adj_dist)

        print('hhh',adj_dist,'ssss',sum1)
        adj_d.append(sparse_mx_to_torch_sparse_tensor(adj_dist).float().cuda())
        # adj_d.append(sparse_mx_to_torch_sparse_tensor(adj_diff).float().cuda())
        adj_base = adj_k1
        # print('low dim index:',i, adj_d[-1])
    adj_L = sp.csr_matrix(adj_L) 
    adj_L = sparse_mx_to_torch_sparse_tensor(adj_L).float().cuda()
    return adj_L, adj_d

def low_dim_all(adj, K, eye):
    """
    adj: the 1-hop normalized adjacency matrix
    K: the most distant node
    eye: Consider eye matrix [self-node] seperately
    """
    adj_K = np.linalg.matrix_power(adj, K)
    adj_L1 = np.array(adj!=0, dtype=np.float32)
    adj_base = adj_L1
    adj_d = []
    if eye:
        eye_adj = np.identity(adj.shape[0])
        if K == 1:
            adj = sp.csr_matrix(adj)
            adj_other = adj_L1 - eye_adj
            eye_adj = eye_adj * adj_K
            adj_other = adj_other * adj_K
            adj_d.append(sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(eye_adj)).float().cuda())
            adj_d.append(sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(adj_other)).float().cuda())
        elif K > 1:
            adj_other = adj_L1 - eye_adj
            eye_adj = eye_adj * adj_K
            adj_other = adj_other * adj_K
            adj_d.append(sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(eye_adj)).float().cuda())
            adj_d.append(sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(adj_other)).float().cuda())
            for i in range(1, K):
                adj_k = np.linalg.matrix_power(adj, i+1)
                adj_k1 = np.array(adj_k!=0, dtype=np.float32)
                adj_diff = adj_k1 - adj_base
                adj_dist = adj_diff * adj_K
                sum1 = np.sum(adj_dist, 1)
                adj_dist = sp.csr_matrix(adj_dist)

                print("The adj of {}th hop".format(i, adj_dist))
                print('sum of this adj:',sum1)
                adj_d.append(sparse_mx_to_torch_sparse_tensor(adj_dist).float().cuda())
                adj_base = adj_k1
    else:
        if K == 1:
            adj_sp = sp.csr_matrix(adj)
            adj_d.append(sparse_mx_to_torch_sparse_tensor(adj_sp).float().cuda())

        elif K > 1:
            adj_sp = sp.csr_matrix(adj)
            adj_d.append(sparse_mx_to_torch_sparse_tensor(adj_sp).float().cuda())
            for i in range(1, K):
                adj_k = np.linalg.matrix_power(adj, i+1)
                adj_k1 = np.array(adj_k!=0, dtype=np.float32)
                adj_diff = adj_k1 - adj_base
                adj_dist = adj_diff * adj_K
                sum1 = np.sum(adj_dist, 1)
                adj_dist = sp.csr_matrix(adj_dist)

                print("The adj of {}th hop".format(i+1, adj_dist))
                print('sum of this adj:',sum1)
                adj_d.append(sparse_mx_to_torch_sparse_tensor(adj_dist).float().cuda())
                adj_base = adj_k1
    return adj_d

def load_citation(dataset_str="cora", normalization="FirstOrderGCN", cuda=True, gamma=1, degree=2, L=None, K=None):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    adj, features = preprocess_citation(adj, features, normalization, gamma=gamma)

    adj = adj.toarray()
    if K:
        print('use low dim:')
        adj, adj_dist = low_dim(adj, L, K)
    else:
        adj = sp.csr_matrix(np.linalg.matrix_power(adj, degree))
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        if cuda:
            adj = adj.cuda()
        adj_dist = []

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, adj_dist, features, labels, idx_train, idx_val, idx_test

def sgc_precompute(features, adj, adj_dist, degree, concat, L, K, idx_train, idx_val, idx_test):
    t = perf_counter()
    mem = [features]
    # for i in range(degree):
    #     features = torch.spmm(adj.cuda(), features)
    #     adj1 = torch.spmm(adj2, adj1)
    #     mem.append(features)
    if K:
        local_features = torch.spmm(adj, features)
        all_features = [[],[],[]]
        
        all_features[0].append(local_features[idx_train])
        all_features[1].append(local_features[idx_val])
        all_features[2].append(local_features[idx_test])

        zz = 0
        for i in range(0, K-L):
            low_feat = torch.spmm(adj_dist[i], features)
            # Check the number of neighbors
            train = low_feat[:120]
            val = low_feat[120:620]
            test = low_feat[-1000:]
            gt1 = torch.sum(train.gt(0))
            te1 = torch.sum(test.gt(0))
            val = torch.sum(val.gt(0))
            print('partion: train | test | val:',gt1/120,te1/1000,val/500)
            # print('total',i,low_feat,low_feat.shape)
            # print('train total:',train,train.shape)
            # print('test total:',test,test.shape)
            all_features[0].append(low_feat[idx_train])
            all_features[1].append(low_feat[idx_val])
            all_features[2].append(low_feat[idx_test])
            zz += low_feat
        # print('total length:',len(all_features))
    else:
        all_features = torch.spmm(adj, features)

    
    precompute_time = perf_counter()-t
    return all_features, precompute_time

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_data(data_path="data/", normalization="AugNormAdj", cuda=True, gamma=1.0):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T + sp.eye(adj.shape[0])
    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    adj_normalizer = fetch_normalization(normalization)
    if "Aug" in normalization:
        adj = adj_normalizer(adj, gamma)
    else:
        adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    if "Aug" in normalization:
        train_adj = adj_normalizer(train_adj, gamma)
    else:
        train_adj = adj_normalizer(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, train_adj, features, labels, train_index, val_index, test_index