This repo is heavily based on the repo for the paper [FLAG: Adversarial Data Augmentation for Graph Neural Networks](https://arxiv.org/abs/2010.09891).

We update the original implementation to icnclude Bayesian Optimization and a Generalized Min-Max-Sum Readout Function.

To run the DeeperGCN variants with the above augmentations, go to the subfolder **6.883GNNOGB/NAS/deep_gcns_torch/examples/ogb/ogbg_mol/** and run the following command:

	python main.py --use_gpu --conv_encode_edge --num_layers 7 --dataset ogbg-molhiv --block res+ --gcn_aggr softmax --t 1.0 --learn_t --dropout 0.5 --step-size 1e-2 --graph_pooling gen
	
Modify the main.py file to specify the parameters for bayesian optimization, and augment the dropout rate and num layers accordingly.
