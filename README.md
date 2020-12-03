# 6.883GNNOGB

This repo contains our final project for 6.883 Meta Learning, Fall 2020


## Dependencies
* python 3.6.9
* pytorch + cuda:  1.4.0+cu101 10.1
* OGB:  1.2.3
* torch_geometric:  1.6.1
* numpy:  1.18.5
* pandas:  1.1.2
* urllib3:  1.24.3
* outdated:  0.2.0

## DeeperGCN Variants
This repo includes scripts to run DeeperGCN with FLAG, Virtual Node, Bayesian Optimization, and a Generalized Readout Function. Details on running with these augmentations and frameworks can be found in the DGCN subfolder

## GraphNAS for OGB
This repo also includes scripts to run a modified GraphNAS framework for OGB graph property prediction tasks. We augment the existing GraphNAS framework to include edge features in the message passing network, add a global mean pooling readout function, and include OGB atom and bond encoders. Details for running NAS can be found in the NAS subfolder
