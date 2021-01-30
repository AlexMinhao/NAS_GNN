import time
import argparse
import numpy as np
import pickle as pkl
import os
from math import log
from citation import train_regression, train_gcn
from models import get_model
from utils import sgc_precompute, load_citation, set_seed
from args import get_citation_args
import torch
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Arguments
args = get_citation_args()

# setting random seeds
set_seed(args.seed, args.cuda)

# Hyperparameter optimization
space = {'weight_decay' : hp.loguniform('weight_decay', log(1e-10), log(1e-4))}

adj, adj_dist, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda, gamma=args.gamma,degree=args.degree, L=args.L, K=args.K)
if args.model != "GCN":
    features, precompute_time = sgc_precompute(features, adj, adj_dist, args.degree, args.concat, args.L, args.K, idx_train, idx_val, idx_test)
def sgc_objective(space):
    if args.K:
        model = get_model(args.model, features[0][0].size(1), labels.max().item()+1, args.hidden, args.decay, args.L, args.K, args.dropout, args.cuda)
    else:
        model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.decay, args.L, args.K, args.dropout, args.cuda)
    if args.model != 'GCN':
        if args.K:
            model, acc_val, _, _ = train_regression(model, features[0], labels[idx_train], features[1], labels[idx_val],  features[2], labels[idx_test], idx_test, adj,
                                      args.epochs, space['weight_decay'], args.lr, args.dropout)
        else:
            model, acc_val, _, _ = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],  features[idx_test], labels[idx_test], idx_test, adj,
                                      args.epochs, space['weight_decay'], args.lr, args.dropout)
    else:
        model, acc_val, _ = train_gcn(model, features, labels, idx_train, idx_val, idx_test, adj,
                        args.epochs, args.weight_decay, args.lr, args.dropout)

    print('weight decay: {:.2e} '.format(space['weight_decay']) + 'accuracy: {:.4f}'.format(acc_val))
    return {'loss': -acc_val, 'status': STATUS_OK}

best = fmin(sgc_objective, space=space, algo=tpe.suggest, max_evals=200)
print("Best weight decay: {:.2e}".format(best["weight_decay"]))
if args.K:
    os.makedirs("tuning/{}-L{}-K{}-d{}-{}-tuning".format(args.model, args.L, args.K, args.decay, args.name), exist_ok=True)
    path = 'tuning/{}-L{}-K{}-d{}-{}-tuning/{}.txt'.format(args.model, args.L, args.K, args.decay, args.name, args.dataset)
else:
    os.makedirs("tuning/{}-{}-tuning".format(args.model,args.degree), exist_ok=True)
    path = 'tuning/{}-{}-tuning/{}.txt'.format(args.model, args.degree, args.dataset)

with open(path, 'wb') as f: pkl.dump(best, f)
print('save in :',path)
