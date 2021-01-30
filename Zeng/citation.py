import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter

from eval_tool import *

# Arguments
args = get_citation_args()

if args.tuned:
    if args.K:
        with open("tuning/{}-L{}-K{}-d{}-{}-tuning/{}.txt".format(args.model, args.L, args.K, args.decay, args.name, args.dataset), 'rb') as f:
            args.weight_decay = pkl.load(f)['weight_decay']
            print("using tuned weight decay: {}".format(args.weight_decay))
    else:
        with open("tuning/{}-{}-tuning/{}.txt".format(args.model, args.degree, args.dataset), 'rb') as f:
            args.weight_decay = pkl.load(f)['weight_decay']
            print("using tuned weight decay: {}".format(args.weight_decay))
else:
    print("Weight Decay: {}".format(args.weight_decay))

# setting random seeds
set_seed(args.seed, args.cuda)

adj, adj_dist, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda, gamma=args.gamma, degree=args.degree, L=args.L, K=args.K)

if args.model != "GCN":
    features, precompute_time = sgc_precompute(features, adj, adj_dist, args.degree, args.concat, args.L, args.K, idx_train, idx_val, idx_test)
    print("{:.4f}s".format(precompute_time))
if args.K:
    model = get_model(args.model, features[0][0].size(1), labels.max().item()+1, args.hidden, args.decay, args.L, args.K, args.dropout, args.cuda)
else:
    model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)

def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels, test_features, test_labels, idx_test, adj,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    eval_best = 0
    test_eval = 0
    test_best = 0
    macro_best = 0
    macro_eval = 0
    micro_eval = 0
    micro_best = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        acc_train = accuracy(output, train_labels)
        # print('aaaa',output.shape,train_features.shape,train_labels.shape)
        # mad = mad_value(output, )
        loss_train.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            output = model(val_features)
            acc_val = accuracy(output, val_labels)
            acc_test = accuracy(model(test_features), test_labels)
            # acc_test, macro, micro = test_func(model, test_features, test_labels, idx_test)
            if eval_best < acc_val:
                eval_best = acc_val
                test_eval = acc_test
                # micro_eval = micro
                # macro_eval = macro
            if test_best < acc_test:
                test_best = acc_test
                # micro_best = micro
                # macro_best = macro
            # print('epoch: {} |loss: {:.4f} |acc: {:.4f}|eval acc: {:.4f}|test acc: {:.4f}| eval best: {:.4f}| eval test: {:.4f}| eval macro: {:.4f}| : {:.4f}| eval micro: test best: {:.4f} | macro best: {:.4f} | micro best: {:.4f}'.format(epoch, loss_train, acc_train, acc_val, acc_test, eval_best, test_eval, macro_eval, micro_eval, test_best, macro_best, micro_best))

    train_time = perf_counter()-t
    return model, eval_best, test_eval, train_time

def test_regression(model, test_features, test_labels):
    model.eval()
    return accuracy(model(test_features), test_labels)

def test_func(model, test_features, test_labels, idx_test):
    model.eval()
    output = model(test_features)
    out_Acc = accuracy(output, test_labels)
    label_max = []
    test_true = np.empty((0))
    test_true = np.append(test_true, test_labels.cpu().numpy(), axis=0)

    for idx in idx_test:
        label_max.append(torch.argmax(output).item())
    test_pred = label_max
    confusion = confusion_matrix(test_true, test_pred)
    confusion_np = np.array(confusion, dtype=float)
    print(confusion_np)

    labelcpu = labels[idx_test].data.cpu()
    macro_f1 = f1_score(labelcpu, label_max, average='macro')
    micro_f1 = f1_score(labelcpu, label_max, average='micro')
    return out_Acc, macro_f1.item(), micro_f1.item()

    
def test_gcn(model, features, labels, idx_test, adj):
    model.eval()
    output = model(features, adj)
    out_Acc = accuracy(output[idx_test], labels[idx_test])
    label_max = []
    test_true = np.empty((0))
    test_true = np.append(test_true, labels[idx_test].cpu().numpy(), axis=0)

    for idx in idx_test:
        label_max.append(torch.argmax(output[idx]).item())
    test_pred = label_max
    confusion = confusion_matrix(test_true, test_pred)
    confusion_np = np.array(confusion, dtype=float)
    print(confusion_np)

    labelcpu = labels[idx_test].data.cpu()
    macro_f1 = f1_score(labelcpu, label_max, average='macro')
    micro_f1 = f1_score(labelcpu, label_max, average='micro')
    return out_Acc, macro_f1.item(), micro_f1.item()

def train_gcn(model, features, labels, idx_train, idx_valid, idx_test, adj,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):

    # optimizer = optim.Adam(model.parameters(), lr=lr,
    #                        weight_decay=weight_decay)
    optimizer = torch.optim.Adam([
        dict(params=model.gc1.parameters(), weight_decay=weight_decay),
        dict(params=model.gc2.parameters(), weight_decay=0)], lr=args.lr)  # GCN: Only perform weight-decay on first convolution.
    t = perf_counter()
    val_max = 0.0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)

        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        print('loss:',loss_train.item())
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            output = model(features, adj)
            acc_val = accuracy(output[idx_valid], labels[idx_valid])
            if acc_val > val_max:
                val_max = acc_val
                out_Acc, macro_f1, micro_f1 = test_gcn(model, features, labels, idx_test, adj)
                print('best acc in val: {:.4f} Test Accuracy: {:.4f} Test macro: {:.4f}, micro: {:.4f}'.format(val_max, out_Acc, macro_f1, micro_f1))
    train_time = perf_counter()-t
    return model, acc_val, train_time

if args.model != "GCN":
    if args.K:
        model, acc_val, acc_test, train_time = train_regression(model, features[0], labels[idx_train], features[1], labels[idx_val], features[2], labels[idx_test], idx_test, adj,
                        args.epochs, args.weight_decay, args.lr, args.dropout)
        # acc_test = test_regression(model, features[2], labels[idx_test])
    else:
        model, acc_val, acc_test, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val], features[idx_test], labels[idx_test], adj,
                        idx_test, args.epochs, args.weight_decay, args.lr, args.dropout)
        # acc_test = test_regression(model, features[idx_test], labels[idx_test])
else:
    model, acc_val, train_time = train_gcn(model, features, labels, idx_train, idx_val, idx_test, adj,
                        args.epochs, args.weight_decay, args.lr, args.dropout)
    acc_test, macro, micro = test_gcn(model, features, labels, idx_test, adj)
    print('Test macro: {:.4f}, micro: {:.4f} '.format(macro,micro))
print("Validation Accuracy: {:.4f} final Test Accuracy: {:.4f}".format(acc_val, acc_test))
print("train time: {:.4f}s".format(train_time))
