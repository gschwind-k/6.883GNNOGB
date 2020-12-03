import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from model import DeeperGCN
from tqdm import tqdm
from args import ArgsInit
from utils.ckpt_util import save_ckpt
import logging
import time
import statistics
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

import pdb
import sys
sys.path.insert(0,'..')
from attacks import *

# BAYESIAN OPTIMIZATION
from bayes_opt import BayesianOptimization
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events


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

def train_flag(model, device, loader, optimizer, task_type, args):
    loss_list = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            is_labeled = batch.y == batch.y
            forward = lambda perturb : model(batch, perturb).to(torch.float32)[is_labeled]
            model_forward = (model, forward)
            target = batch.y.to(torch.float32)[is_labeled]
            perturb_shape = (batch.x.shape[0], args.hidden_channels)

            if "classification" in task_type:
                criterion = cls_criterion
            else:
                criterion = reg_criterion

            loss, out = flag(model_forward, perturb_shape, target, args, optimizer, device, criterion)

            loss_list.append(loss.item())
    return statistics.mean(loss_list)


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

    return evaluator.eval(input_dict)


def main(lr, dropout):

    args = ArgsInit(lr,dropout).save_exp()

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    sub_dir = 'BS_{}-NF_{}'.format(args.batch_size,
                                   args.feature)

    dataset = PygGraphPropPredDataset(name=args.dataset)
    args.num_tasks = dataset.num_tasks
    logging.info('%s' % args)

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

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    results = {'highest_valid': 0,
               'final_train': 0,
               'final_test': 0,
               'highest_train': 0}

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        logging.info("=====Epoch {}".format(epoch))
        logging.info('Training...')

        # epoch_loss = train(model, device, train_loader, optimizer, dataset.task_type)
        epoch_loss = train_flag(model, device, train_loader, optimizer, dataset.task_type, args)

        logging.info('Evaluating...')
        train_result = eval(model, device, train_loader, evaluator)[dataset.eval_metric]
        valid_result = eval(model, device, valid_loader, evaluator)[dataset.eval_metric]
        test_result = eval(model, device, test_loader, evaluator)[dataset.eval_metric]

        logging.info({'Train': train_result,
                      'Validation': valid_result,
                      'Test': test_result})

        model.print_params(epoch=epoch)

        if train_result > results['highest_train']:

            results['highest_train'] = train_result

        if valid_result > results['highest_valid']:
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result

            # save_ckpt(model, optimizer,
            #           round(epoch_loss, 4), epoch,
            #           args.model_save_path,
            #           sub_dir, name_post='valid_best')

    logging.info("%s" % results)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))

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


if __name__ == "__main__":
    cls_criterion = torch.nn.BCEWithLogitsLoss()
    reg_criterion = torch.nn.MSELoss()

    #pbounds = {'learning_rate': (.001, .1), 'dropout': (.3, .9)}
    #bayesian_optimization(f=main, p_bounds=pbounds)
    main(lr=0.01, dropout=0.5)
