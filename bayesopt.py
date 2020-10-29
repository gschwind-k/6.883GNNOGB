import argparse
import uuid
import logging
import time
import os
import sys
import glob
import shutil
from tqdm import tqdm
import numpy as np
import statistics
from collections import OrderedDict
from sklearn.metrics import roc_auc_score

# PYTORCH PACKAGES
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter, scatter_softmax
from torch_geometric.utils import degree
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import torch.optim as optim

# BAYESIAN OPTIMIZATION
from bayes_opt import BayesianOptimization
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

# DEEPERGCN MODEL
from deeperGCN import *

from args import ArgsInit

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


# training loop
def train(model, device, loader, optimizer, task_type):
    loss_list = []
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            pred = model(batch)
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
    return statistics.mean(loss_list)

# evaluation function
@torch.no_grad()
def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            pred = model(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true,
                  "y_pred": y_pred}

    print("ROC-AUC: ", roc_auc_score(y_true, y_pred))
    return evaluator.eval(input_dict)

# save pytorch model checkpoints
def save_ckpt(model, optimizer, loss, epoch, save_path, name_pre, name_post='best'):
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    state = {
            'epoch': epoch,
            'model_state_dict': model_cpu,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Directory ", save_path, " is created.")

    filename = '{}/{}_{}.pth'.format(save_path, name_pre, name_post)
    torch.save(state, filename)
    print('model has been saved as {}'.format(filename))

# runs DeeperGCN model for given number of epochs with provided parameters
def m(learning_rate, dropout, batch_size=64, description='DeeperGCN small', epochs=1,  
        num_layers=7, mlp_layers=1, hidden_channels=256, gcn_aggr='max', 
       graph_pooling='mean', experiment_save="bench"):
    # args = ArgsInit() #.save_exp()

    args = ArgsInit(description=description, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, 
       dropout=dropout, num_layers=num_layers, mlp_layers=mlp_layers, hidden_channels=hidden_channels,
       gcn_aggr=gcn_aggr, graph_pooling=graph_pooling, experiment_save=experiment_save)
    print("args ", args)

    device = torch.device("cpu")

    sub_dir = 'BS_{}-NF_{}'.format(args.batch_size,
                                   args.feature)

    dataset = PygGraphPropPredDataset(name=args.dataset)
    args.num_tasks = dataset.num_tasks
    logging.info('%s' % args)
    print('%s' % args)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    evaluator = Evaluator(args.dataset)
    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = DeeperGCN(args).to(device)

    logging.info(model)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    results = {'highest_valid': 0,
               'final_train': 0,
               'final_test': 0,
               'highest_train': 0}

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        logging.info("=====Epoch {}".format(epoch))
        logging.info('Training...')
        print("=====Epoch {}".format(epoch))
        print("Training...")

        epoch_loss = train(model, device, train_loader, optimizer, dataset.task_type)

        logging.info('Evaluating...')
        print('Evaluating...')
        train_result = eval(model, device, train_loader, evaluator)[dataset.eval_metric]
        valid_result = eval(model, device, valid_loader, evaluator)[dataset.eval_metric]
        test_result = eval(model, device, test_loader, evaluator)[dataset.eval_metric]

        logging.info({'Train': train_result,
                      'Validation': valid_result,
                      'Test': test_result})
        print({'Train': train_result,
                      'Validation': valid_result,
                      'Test': test_result})


        model.print_params(epoch=epoch)

        if train_result > results['highest_train']:

            results['highest_train'] = train_result

        if valid_result > results['highest_valid']:
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result

            save_ckpt(model, optimizer,
                      round(epoch_loss, 4), epoch,
                      args.model_save_path,
                      sub_dir, name_post='valid_best')

    logging.info("%s" % results)
    print("%s" % results)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))
    print('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))
    assert False
    return results['final_test']



def bayesian_optimization(f, p_bounds, init_points=2, n_iter=10):
     
    optimizer = BayesianOptimization(
        f=f,
        pbounds=pbounds,
        random_state=1,
    )

    logger = JSONLogger(path = "./logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=2,
        n_iter=10,
    )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print("Max:", optimizer.max)



cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

if __name__ == "__main__":
    pbounds = {'learning_rate': (.001, .1), 'dropout': (.1, .9)}
    bayesian_optimization(f=m, p_bounds=pbounds)