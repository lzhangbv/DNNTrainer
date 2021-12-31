# Distributed GNN Training

## Partitioning-based

* Neugraph: parallel deep neural network computation on large graphs (ATC 2019)
* Improving the accuracy, scalability, and performance of graph neural networks with roc (MLSys 2020)
* DGCL: an efficient communication library for distributed GNN training (EuroSys 2021)
* P3: Distributed Deep Graph Learning at Scale (OSDI 2021)

## Sampling-based

* Fast Graph Representation Learning with PyTorch Geometric (PyG)
* Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs (DGL) 
* PaGraph: Scaling GNN Training on Large Graphs via Computation-aware Caching (SoCC 2020)
* BGL: GPU-Efficient GNN Training by Optimizing Graph Data I/O and Preprocessing (ByteDance)

We are interested in sampling-based GNN training, with a special focus on IO performance. Therefore, popular GNN training frameworks like DGL and PyG are used to illustrate the distributed GNN training workflow. 