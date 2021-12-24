import os
import re
import warnings

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from ..pt_overrides.override_data_parallel import LightningDistributedDataParallel

"""
We only focus on DDP but ignore DP and AMP training and callbacks.
Usage:
    trainer = Trainer(gpus=[0,1,2,3], nb_gpu_nodes=2) 
fit -> ddp_train:
    (1) mp.spawn
    (2) dist init
    (3) override root_module for ddp forward pass
    (4) get_model returns the root_module model
"""

class Trainer(TrainerIO):

    def __init__(self,
                 experiment=None, cluster=None,
                 max_nb_epochs=1000,
                 nb_gpu_nodes=1,
                 gpus=[0]):
        # Transfer params
        self.experiment = experiment
        self.cluster = cluster
        self.max_nb_epochs = max_nb_epochs

        # Multi-node Multi-GPU settings
        self.nb_gpu_nodes = nb_gpu_nodes
        self.data_parallel_device_ids = gpus
        self.use_ddp = len(self.data_parallel_device_ids) * nb_gpu_nodes > 1

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in self.data_parallel_device_ids])
        print('VISIBLE GPUS: %r' % os.environ["CUDA_VISIBLE_DEVICES"])

    def __get_model(self):
        """get root_module model"""
        return self.model.module if self.use_ddp else self.model

    # -----------------------------
    # MODEL TRAINING
    # logic: fit -> __single_gpu_train or __ddp_train -> __run_pretrain_routine -> __train -> __run_tng_batch
    # -----------------------------
    def fit(self, model):
        """single_gpu_train or ddp_train"""
        if self.use_ddp:
            # launch multiple processes (per-gpu per-process)
            # when using multi-node: each node should start the module
            mp.spawn(self.__ddp_train, nprocs=len(self.data_parallel_device_ids), args=(model, ))
        else:
            self.__single_gpu_train(model)
        return 1

    def __single_gpu_train(self, model):
        # ...
        model.cuda(self.data_parallel_device_ids[0])
        self.__run_pretrain_routine(model)

    def __ddp_train(self, gpu_nb, model):
        # determine which process we are and world size
        self.proc_rank = self.node_rank * len(self.data_parallel_device_ids) + gpu_nb
        self.world_size = self.nb_gpu_nodes * len(self.data_parallel_device_ids)

        # set up server using proc 0's ip address
        self.__init_tcp_connection()

        torch.cuda.set_device(gpu_nb)
        model.cuda(gpu_nb)

        # input model (root_module instance); output model (ddp override instance)
        model = LightningDistributedDataParallel(model, device_ids=[gpu_nb], find_unused_parameters=True)

        # continue training routine
        self.__run_pretrain_routine(model)

    def __init_tcp_connection(self):
        """
        dist.init_process_group()
        """
        # sets the appropriate port
        try:
            port = os.environ['MASTER_PORT']
        except Exception:
            port = 12910
            os.environ['MASTER_PORT'] = str(port)

        # figure out the root node addr
        try:
            root_node = os.environ['SLURM_NODELIST'].split(' ')[0]
        except Exception:
            root_node = '127.0.0.2'

        root_node = self.resolve_root_node_address(root_node)
        os.environ['MASTER_ADDR'] = root_node

        dist.init_process_group("nccl", rank=self.proc_rank, world_size=self.world_size)

    def resolve_root_node_address(self, root_node):
        if '[' in root_node:
            name = root_node.split('[')[0]
            number = root_node.split(',')[0]
            if '-' in number:
                number = number.split('-')[0]

            number = re.sub('[^0-9]', '', number)
            root_node = name + number

        return root_node

    def __run_pretrain_routine(self, model):
        """
        prepare a few things before starting actual training
        """
        self.model = model

        # pt_model (root_module instance)
        pt_model = self.__get_model()

        # prepare dataloaders and optimizers
        self.get_dataloaders(pt_model)
        self.optimizers = pt_model.configure_optimizers()
        if len(self.optimizers) == 2:
            self.optimizers, self.lr_schedulers = self.optimizers

        self.__train()

    def __train(self):
        """run all epochs"""
        for epoch_nb in range(self.current_epoch, self.max_nb_epochs):
            # ...
            for batch_nb, data_batch in enumerate(self.tng_dataloader):
                # ...
                batch_result = self.__run_tng_batch(data_batch, batch_nb)
                # ...
                
    def __run_tng_batch(self, data_batch, batch_nb):
        # forward pass
        if self.use_ddp:
            output = self.model(data_batch, batch_nb)
        else:
            output = self.model.training_step(data_batch, batch_nb)
        loss = output['loss']

        # backward pass
        loss.backward()
        #...