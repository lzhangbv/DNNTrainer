import os
import torch
import math

from .memory import ModelSummary
from .grads import GradInformation
from .model_saving import ModelIO
from .optimization import OptimizerConfig
from .hooks import ModelHooks


"""
    RootModule (NotImplemented):
        - model: forward
        - datasets: tng_dataloader, test_dataloader, val_dataloader
        - optimizer: configure_optimizers
        - units: training_step, validation_step, validation_end, 
    ModelIO (NotImplemented):
        load_model_specific, get_save_dict
    GradInfomation:
        grad_norm, describe_grads, describe_params
    OptimizerConfig:
        choose_optimizer
    ModelHooks (optional):
        on_batch_start, on_batch_end, on_epoch_start, on_epoch_end
"""

class RootModule(ModelIO, GradInformation, OptimizerConfig, ModelHooks):

    def __init__(self, hparams):
        super(RootModule, self).__init__()
        self.hparams = hparams
        self.on_gpu = hparams.on_gpu
        self.dtype = torch.FloatTensor
        self.exp_save_path = None
        self.current_epoch = 0
        self.global_step = 0
        self.loaded_optimizer_states_dict = {}
        self.num = 2

        # computed vars for the dataloaders
        self._tng_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

        if self.on_gpu:
            print('running on gpu...')
            self.dtype = torch.cuda.FloatTensor
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

    def forward(self, *args, **kwargs):
        """
        Expand model in into whatever you need.
        Also need to return the target
        :param x:
        :return:
        """
        raise NotImplementedError

    def validation_step(self, data_batch):
        """
        return whatever outputs will need to be aggregated in validation_end
        :param data_batch:
        :return:
        """
        raise NotImplementedError

    def validation_end(self, outputs):
        """
        Outputs has the appended output after each validation step
        :param outputs:
        :return: dic_with_metrics for tqdm
        """
        raise NotImplementedError

    def training_step(self, data_batch):
        """
        return loss, dict with metrics for tqdm
        :param data_batch:
        :return:
        """
        raise NotImplementedError

    def configure_optimizers(self):
        """
        Return array of optimizers
        :return:
        """
        raise NotImplementedError

    def update_tng_log_metrics(self, logs):
        """
        Chance to update metrics to be logged for training step.
        For example, add music, images, etc... to log
        :param logs:
        :return:
        """
        raise NotImplementedError

    def summarize(self):
        model_summary = ModelSummary(self)
        print(model_summary)

    @property
    def tng_dataloader(self):
        """
        Implement a function to load an h5py of this data
        :return:
        """
        raise NotImplementedError

    @property
    def test_dataloader(self):
        """
        Implement a function to load an h5py of this data
        :return:
        """
        raise NotImplementedError

    @property
    def val_dataloader(self):
        """
        Implement a function to load an h5py of this data
        :return:
        """
        raise NotImplementedError
