This repo draws heavily from the original repo for GraphNAS: https://github.com/GraphNAS/GraphNAS

# GraphNAS

#### Overview
Graph Neural Architecture Search (GraphNAS for short) enables automatic design of the best graph neural architecture 
based on reinforcement learning. 

We augment the implementation of GraphNAS to include edge features in the message passing network, as well as a global pooling layer and input encoders for graph property prediction tasks. We also include data loaders for OGB benchmark tasks.

To run our modified NAS search on the ogb-molhiv dataset, run the following command:

    python -m graphnas.main --dataset ogb
