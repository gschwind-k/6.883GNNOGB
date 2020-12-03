import torch
import torch.nn.functional as F

from graphnas.gnn import GraphNet as BaseNet
from graphnas.search_space import act_map
from graphnas_variants.macro_graphnas.pyg.pyg_gnn_layer import GeoLayer
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool



class GraphNet(BaseNet):

    def __init__(self, actions, num_feat, num_label, drop_out=0.6, multi_label=False, batch_normal=True, state_num=5,
                 residual=False):
        self.residual = residual
        self.batch_normal = batch_normal
        super(GraphNet, self).__init__(actions, num_feat, num_label, drop_out, multi_label, batch_normal, residual,
                                       state_num)

        self.atom_encoder = AtomEncoder(emb_dim=256)
        self.edge_encoder = BondEncoder(emb_dim=256)
        self.graph_pred_linear = torch.nn.Linear(256, 1)

    def build_model(self, actions, batch_normal, drop_out, num_feat, num_label, state_num):
        if self.residual:
            self.fcs = torch.nn.ModuleList()
        if self.batch_normal:
            self.bns = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.acts = []
        self.gates = torch.nn.ModuleList()
        self.build_hidden_layers(actions, batch_normal, drop_out, self.layer_nums, num_feat, num_label, state_num)

    def build_hidden_layers(self, actions, batch_normal, drop_out, layer_nums, num_feat, num_label, state_num=6):

        # build hidden layer
        for i in range(layer_nums):

            if i == 0:
                in_channels = 256 #change this to embded_dim?
            else:
                in_channels = out_channels * head_num

            # extract layer information
            attention_type = actions[i * state_num + 0]
            aggregator_type = actions[i * state_num + 1]
            act = actions[i * state_num + 2]
            head_num = actions[i * state_num + 3]
            out_channels = actions[i * state_num + 4]
            concat = True
            if i == layer_nums - 1:
                concat = False
            if self.batch_normal:
                self.bns.append(torch.nn.BatchNorm1d(in_channels, momentum=0.5))
            self.layers.append(
                GeoLayer(in_channels, out_channels, head_num, concat, dropout=self.dropout,
                         att_type=attention_type, agg_type=aggregator_type, ))
            self.acts.append(act_map(act))
            if self.residual:
                if concat:
                    self.fcs.append(torch.nn.Linear(in_channels, out_channels * head_num))
                else:
                    self.fcs.append(torch.nn.Linear(in_channels, out_channels))

    def forward(self, batch):
        x = batch.x
        edge_index_all = batch.edge_index
        edge_attr = batch.edge_attr
        #encode atom features
        output = self.atom_encoder(x)
        #encode edge features
        edge_embed = self.edge_encoder(edge_attr)

        output = output.type(torch.FloatTensor).to(x.device)
        if self.residual:
            for i, (act, layer, fc) in enumerate(zip(self.acts, self.layers, self.fcs)):
                output = layer(output, edge_index_all, edge_embed) #pass through layer
                output = act(output + fc(output)) #non linearity + residual
                output = F.dropout(output, p=self.dropout, training = self.training) # dropout layer
                if self.batch_normal:
                    output = self.bns[i](output)
        else:
            for i, (act, layer) in enumerate(zip(self.acts, self.layers)):
                output = act(layer(output, edge_index_all, edge_embed))
                output = F.dropout(output, p=self.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)

        h_graph = global_mean_pool(output, batch.batch) #batch_wise graph level outputs
        return self.graph_pred_linear(h_graph) #return batch_wise predictions
        
    def __repr__(self):
        result_lines = ""
        for each in self.layers:
            result_lines += str(each)
        return result_lines

    @staticmethod
    def merge_param(old_param, new_param, update_all):
        for key in new_param:
            if update_all or key not in old_param:
                old_param[key] = new_param[key]
        return old_param

    def get_param_dict(self, old_param=None, update_all=True):
        if old_param is None:
            result = {}
        else:
            result = old_param
        for i in range(self.layer_nums):
            key = "layer_%d" % i
            new_param = self.layers[i].get_param_dict()
            if key in result:
                new_param = self.merge_param(result[key], new_param, update_all)
                result[key] = new_param
            else:
                result[key] = new_param
        if self.residual:
            for i, fc in enumerate(self.fcs):
                key = f"layer_{i}_fc_{fc.weight.size(0)}_{fc.weight.size(1)}"
                result[key] = self.fcs[i]
        if self.batch_normal:
            for i, bn in enumerate(self.bns):
                key = f"layer_{i}_fc_{bn.weight.size(0)}"
                result[key] = self.bns[i]
        return result

    def load_param(self, param):
        if param is None:
            return

        for i in range(self.layer_nums):
            self.layers[i].load_param(param["layer_%d" % i])

        if self.residual:
            for i, fc in enumerate(self.fcs):
                key = f"layer_{i}_fc_{fc.weight.size(0)}_{fc.weight.size(1)}"
                if key in param:
                    self.fcs[i] = param[key]
        if self.batch_normal:
            for i, bn in enumerate(self.bns):
                key = f"layer_{i}_fc_{bn.weight.size(0)}"
                if key in param:
                    self.bns[i] = param[key]
