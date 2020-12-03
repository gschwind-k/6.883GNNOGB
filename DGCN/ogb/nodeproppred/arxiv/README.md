# ogbn-arxiv

To train baselines with FLAG in the default setup, run

**MLP+FLAG**
                    
        python mlp.py

**GCN+FLAG**

        python gnn.py

**GraphSAGE+FLAG**
        
        python gnn.py --use_sage

**GAT+FLAG**

For baseline GAT model, please refer to [here](https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv). This one is the `GAT(norm. adj.)+labels` version.
        
        python gat_dgl/gat.py --use-norm --use-labels
