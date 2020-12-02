import os.path as osp
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Amazon

# ogb data loaders
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from ogb.graphproppred import Evaluator

from graphnas.gnn_model_manager import CitationGNNManager, evaluate
from graphnas_variants.macro_graphnas.pyg.pyg_gnn import GraphNet
from graphnas.utils.label_split import fix_size_split

from sklearn.metrics import roc_auc_score

from graphnas.utils.model_utils import EarlyStop, TopAverage, process_action
import statistics

def load_data(dataset="ogb", supervised=False, full_data=True):
    '''
    support semi-supervised and supervised
    :param dataset:
    :param supervised:
    :return:
    '''
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    # if dataset == 'ogb':
    dataset = PygGraphPropPredDataset(name = "ogbg-molhiv")

    # elif dataset in ["CS", "Physics"]:
    #     dataset = Coauthor(path, dataset, T.NormalizeFeatures())
    # elif dataset in ["Computers", "Photo"]:
    #     dataset = Amazon(path, dataset, T.NormalizeFeatures())
    # elif dataset in ["Cora", "Citeseer", "Pubmed"]:
    #     dataset = Planetoid(path, dataset, "public")
    # data = dataset[0]
    # if supervised:
    #     if full_data:
    #         data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
    #         data.train_mask[:-1000] = 1
    #         data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
    #         data.val_mask[data.num_nodes - 1000: data.num_nodes - 500] = 1
    #         data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
    #         data.test_mask[data.num_nodes - 500:] = 1
    #     else:
    #         data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
    #         data.train_mask[:1000] = 1
    #         data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
    #         data.val_mask[data.num_nodes - 1000: data.num_nodes - 500] = 1
    #         data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
    #         data.test_mask[data.num_nodes - 500:] = 1
    return dataset


# class GeoCitationManager(CitationGNNManager):
#     def __init__(self, args):
#         super(GeoCitationManager, self).__init__(args)
#         if hasattr(args, "supervised"):
#             self.data = load_data(args.dataset, args.supervised)
#         else:
#             self.data = load_data(args.dataset)
#         self.args.in_feats = self.in_feats = self.data.num_features
#         self.args.num_class = self.n_classes = self.data.y.max().item() + 1
#         device = torch.device('cuda' if args.cuda else 'cpu')
#         self.data.to(device)

#     def build_gnn(self, actions):
#         model = GraphNet(actions, self.in_feats, self.n_classes, drop_out=self.args.in_drop, multi_label=False,
#                          batch_normal=False, residual=False)
#         return model

#     def update_args(self, args):
#         self.args = args

#     def save_param(self, model, update_all=False):
#         pass

#     def shuffle_data(self, full_data=True):
#         device = torch.device('cuda' if self.args.cuda else 'cpu')
#         if full_data:
#             self.data = fix_size_split(self.data, self.data.num_nodes - 1000, 500, 500)
#         else:
#             self.data = fix_size_split(self.data, 1000, 500, 500)
#         self.data.to(device)

#     @staticmethod
#     def run_model(model, optimizer, loss_fn, data, epochs, early_stop=5, tmp_model_file="geo_citation.pkl",
#                   half_stop_score=0, return_best=False, cuda=True, need_early_stop=False, show_info=False):

#         dur = []
#         begin_time = time.time()
#         best_performance = 0
#         min_val_loss = float("inf")
#         min_train_loss = float("inf")
#         model_val_acc = 0
#         print("Number of train datas:", data.train_mask.sum())
#         for epoch in range(1, epochs + 1):
#             model.train()
#             t0 = time.time()
#             # forward
#             logits = model(data.x, data.edge_index)
#             logits = F.log_softmax(logits, 1)
#             loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss = loss.item()

#             # evaluate
#             model.eval()
#             logits = model(data.x, data.edge_index)
#             logits = F.log_softmax(logits, 1)
#             train_acc = evaluate(logits, data.y, data.train_mask)
#             dur.append(time.time() - t0)

#             val_acc = evaluate(logits, data.y, data.val_mask)
#             test_acc = evaluate(logits, data.y, data.test_mask)

#             loss = loss_fn(logits[data.val_mask], data.y[data.val_mask])
#             val_loss = loss.item()
#             if val_loss < min_val_loss:  # and train_loss < min_train_loss
#                 min_val_loss = val_loss
#                 min_train_loss = train_loss
#                 model_val_acc = val_acc
#                 if test_acc > best_performance:
#                     best_performance = test_acc
#             if show_info:
#                 print(
#                     "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
#                         epoch, loss.item(), np.mean(dur), train_acc, val_acc, test_acc))

#                 end_time = time.time()
#                 print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))
#         print(f"val_score:{model_val_acc},test_score:{best_performance}")
#         if return_best:
#             return model, model_val_acc, best_performance
#         else:
#             return model, model_val_acc


class GeoOGBManager(CitationGNNManager):
    def __init__(self, args):
        super(GeoOGBManager, self).__init__(args)
        self.data = load_data(args.dataset)
        self.args.in_feats = self.in_feats = self.data.num_features
        self.args.num_class = self.n_classes = self.data.data.y.max().item() + 1
        device = torch.device('cuda' if args.cuda else 'cpu')

    def build_gnn(self, actions):
        model = GraphNet(actions, self.in_feats, self.n_classes, drop_out=self.args.in_drop, multi_label=False,
                         batch_normal=False, residual=False)
        return model

    def update_args(self, args):
        self.args = args

    def save_param(self, model, update_all=False):
        pass

    # i think we don't need this
    # def shuffle_data(self, full_data=True):
    #     device = torch.device('cuda' if self.args.cuda else 'cpu')
    #     if full_data:
    #         self.data = fix_size_split(self.data, self.data.num_nodes - 1000, 500, 500)
    #     else:
    #         self.data = fix_size_split(self.data, 1000, 500, 500)
    #     self.data.to(device)

    def train(self, actions=None, format="two"):
        origin_action = actions
        actions = process_action(actions, format, self.args)
        print("train action:", actions)

        # create model
        model = self.build_gnn(actions)

        try:
            if self.args.cuda:
                model.cuda()
            # use optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            model, val_acc = self.run_model_ogb(model, self.data, optimizer, batch_size=128, epochs=self.args.epochs)
            # model, val_acc = self.run_model(model, optimizer, self.loss_fn, self.data, self.epochs, cuda=self.args.cuda,
            #                                 half_stop_score=max(self.reward_manager.get_top_average() * 0.7, 0.4))
        except RuntimeError as e:
            import traceback

            if "cuda" in str(e) or "CUDA" in str(e):
                traceback.print_exc()
                val_acc = 0
            else:
                raise e
        reward = self.reward_manager.get_reward(val_acc)
        self.save_param(model, update_all=(reward > 0))

        self.record_action_info(origin_action, reward, val_acc)

        return reward, val_acc
    
    def train_ogb(self, model, device, loader, optimizer, task_type):
        loss_list = []
        # loss_item = flag(model, device, loader, optimizer, task_type)
        # loss_list.append(loss_item)
        model.train()
        cls_criterion = torch.nn.BCEWithLogitsLoss()
        reg_criterion = torch.nn.MSELoss()

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
    
    @torch.no_grad()
    def eval(self,model, device, loader, evaluator):
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

    def run_model_ogb(self, model, dataset, optimizer, batch_size=32, epochs=100):
        # args = ArgsInit() #.save_exp()

        device = torch.device("cuda")
        evaluator = Evaluator(name="ogbg-molhiv")

        split_idx = dataset.get_idx_split()

        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)

        results = {'highest_valid': 0,
                   'final_train': 0,
                   'final_test': 0,
                   'highest_train': 0}

        for epoch in range(1, epochs + 1):
            epoch_loss = self.train_ogb(model, device, train_loader, optimizer, dataset.task_type)

            train_result = self.eval(model, device, train_loader, evaluator)[dataset.eval_metric]
            valid_result = self.eval(model, device, valid_loader, evaluator)[dataset.eval_metric]
            test_result = self.eval(model, device, test_loader, evaluator)[dataset.eval_metric]

            print({'Train': train_result,
                          'Validation': valid_result,
                          'Test': test_result})

            if train_result > results['highest_train']:

                results['highest_train'] = train_result

            if valid_result > results['highest_valid']:
                results['highest_valid'] = valid_result
                results['final_train'] = train_result
                results['final_test'] = test_result

        return model, results['final_test']

    # @staticmethod
    # def run_model(model, optimizer, loss_fn, data, epochs, early_stop=5, tmp_model_file="geo_citation.pkl",
    #               half_stop_score=0, return_best=False, cuda=True, need_early_stop=False, show_info=False):

    #     dur = []
    #     begin_time = time.time()
    #     best_performance = 0
    #     min_val_loss = float("inf")
    #     min_train_loss = float("inf")
    #     model_val_acc = 0
    #     # print("Number of train datas:", data.train_mask.sum())
    #     for epoch in range(1, epochs + 1):
    #         model.train()
    #         t0 = time.time()
    #         # forward
    #         logits = model(data.x, data.edge_index)
    #         logits = F.log_softmax(logits, 1)
    #         loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         train_loss = loss.item()

    #         # evaluate
    #         model.eval()
    #         logits = model(data.x, data.edge_index)
    #         logits = F.log_softmax(logits, 1)
    #         train_acc = evaluate(logits, data.y, data.train_mask)
    #         dur.append(time.time() - t0)

    #         val_acc = evaluate(logits, data.y, data.val_mask)
    #         test_acc = evaluate(logits, data.y, data.test_mask)

    #         loss = loss_fn(logits[data.val_mask], data.y[data.val_mask])
    #         val_loss = loss.item()
    #         if val_loss < min_val_loss:  # and train_loss < min_train_loss
    #             min_val_loss = val_loss
    #             min_train_loss = train_loss
    #             model_val_acc = val_acc
    #             if test_acc > best_performance:
    #                 best_performance = test_acc
    #         if show_info:
    #             print(
    #                 "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
    #                     epoch, loss.item(), np.mean(dur), train_acc, val_acc, test_acc))

    #             end_time = time.time()
    #             print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))
    #     print(f"val_score:{model_val_acc},test_score:{best_performance}")
    #     if return_best:
    #         return model, model_val_acc, best_performance
    #     else:
    #         return model, model_val_acc
