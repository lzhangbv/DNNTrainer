import torch
import numpy as np
import traceback
from ..root_module.model_saving import TrainerIO

"""
Support two callbacks in the training progress (i.e., on_epoch_end)
    (1) early stopping: ref to __run_validation
    (2) model checkpoint: ref to __train

APIs:
    fit: train and in-progress validation
    validation: eval
"""

class Trainer(TrainerIO):

    def __init__(self,
                 experiment, cluster, # test_tube required
                 checkpoint_callback, 
                 early_stop_callback, 
                 enable_early_stop=True, 
                 max_nb_epochs=5, 
                 min_nb_epochs=1,
                 current_gpu_name=0,
                 on_gpu=False):

        # Transfer params
        self.experiment = experiment
        self.exp_save_path = experiment.get_data_path(experiment.name, experiment.version)
        self.cluster = cluster

        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_callback.save_function = self.save_checkpoint  # check TrainIO

        self.early_stop = early_stop_callback
        self.early_stop_callback = early_stop_callback
        self.enable_early_stop = enable_early_stop
        self.max_nb_epochs = max_nb_epochs
        self.min_nb_epochs = min_nb_epochs

        self.on_gpu = on_gpu
        self.current_gpu_name = current_gpu_name

        # training state
        self.global_step = 0
        self.current_epoch = 0

        # logging (ignored)

        # dataloaders
        self.tng_dataloader = None
        self.test_dataloader = None
        self.val_dataloader = None

    def __is_function_implemented(self, f_name): 
        # check hooks
        f_op = getattr(self, f_name, None)
        return callable(f_op)

    @property
    def __tng_tqdm_dic(self):
        tqdm_dic = {
            'tng_loss': '{0:.3f}'.format(self.avg_loss),
            'gpu': '{}'.format(self.current_gpu_name),
            'v_nb': '{}'.format(self.experiment.version),
            'epoch': '{}'.format(self.current_epoch),
            'batch_nb':'{}'.format(self.batch_nb),
        }
        tqdm_dic.update(self.tqdm_metrics)
        return tqdm_dic

    def __add_tqdm_metrics(self, metrics):
        for k, v in metrics.items():
            self.tqdm_metrics[k] = v

    def __layout_bookkeeping(self):
        # training bookkeeping
        self.running_loss = []
        self.avg_loss = 0
        self.tqdm_metrics = {}

    def __get_dataloaders(self, model):
        """
        Dataloaders are provided by the model
        """
        self.tng_dataloader = model.tng_dataloader
        self.test_dataloader = model.test_dataloader
        self.val_dataloader = model.val_dataloader

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    def fit(self, model):
        """ 
        Main function
        """
        self.model = model
        model.summarize()

        # transfer dataloaders and optimizers from model
        self.__get_dataloaders(model)
        self.optimizers = model.configure_optimizers()

        # init training constants
        self.__layout_bookkeeping()

        # put on gpu if needed
        if self.on_gpu:
            model = model.cuda()

        # save exp to get started
        self.experiment.save()

        # enable cluster checkpointing
        self.enable_auto_hpc_walltime_manager()  # check TrainIO

        # ---------------------------
        # CORE TRAINING LOOP
        # ---------------------------
        self.__train()

    def __train(self):
        """run train for all epochs (used in fit)"""
        for epoch_nb in range(self.current_epoch, self.max_nb_epochs):
            self.model.current_epoch = epoch_nb

            # on_epoch_start hook
            if self.__is_function_implemented('on_epoch_start'):
                self.model.on_epoch_start()

            self.current_epoch = epoch_nb
            self.batch_loss_value = 0  # accumulated grads

            for batch_nb, data_batch in enumerate(self.tng_dataloader):
                self.global_step += 1
                self.model.global_step = self.global_step

                # on_batch_start hook
                if self.__is_function_implemented('on_batch_start'):
                    self.model.on_batch_start()

                # RUN TRAIN STEP
                self.__run_tng_batch(data_batch)

                # on_batch_end hook
                if self.__is_function_implemented('on_batch_end'):
                    self.model.on_batch_end()
            
            # run validation every epoch
            self.__run_validation()

            # on_epoch_end hook
            if self.__is_function_implemented('on_epoch_end'):
                self.model.on_epoch_end()

            # early stopping
            if self.enable_early_stop:
                should_stop = self.early_stop_callback.on_epoch_end(epoch=epoch_nb, logs=self.__tng_tqdm_dic)
                met_min_epochs = epoch_nb > self.min_nb_epochs

                # stop training
                stop = should_stop and met_min_epochs
                if stop:
                    return

    def __run_tng_batch(self, data_batch):
        """run train for one batch"""
        # forward pass
        # return a scalar value and a dic with tqdm metrics
        loss, model_specific_tqdm_metrics_dic = self.model.training_step(data_batch)
        self.__add_tqdm_metrics(model_specific_tqdm_metrics_dic)

        # backward pass
        loss.backward()
        self.batch_loss_value += loss.item()

        # update gradients across all optimizers
        for optimizer in self.optimizers:
            optimizer.step()

            # clear gradients
            optimizer.zero_grad()

        # track loss
        self.running_loss.append(self.batch_loss_value)
        self.batch_loss_value = 0
        self.avg_loss = np.mean(self.running_loss[-100:])

    def __run_validation(self):
        """run validation for one epoch in the training progress"""    
        # pre hook
        if self.__is_function_implemented('on_pre_performance_check'):
            self.model.on_pre_performance_check()

        # validation
        model_specific_tqdm_metrics_dic = self.validate(
            self.model,
            self.val_dataloader,
            None
        )
        self.__add_tqdm_metrics(model_specific_tqdm_metrics_dic)

        # post hook
        if self.__is_function_implemented('on_post_performance_check'):
            self.model.on_post_performance_check()

        # model checkpointing
        print('save callback...')
        self.checkpoint_callback.on_epoch_end(epoch=self.current_epoch, logs=self.__tng_tqdm_dic)


        # -----------------------------
    
    # -----------------------------
    # MODEL VALIDATION
    # -----------------------------
    def validate(self, model, dataloader, max_batches):
        """ validation API (used in fit or eval)

        for i, data_batch in enumerate(dataloader):
            outputs.append(model.validation_step(data_batch))
            val_results = model.validation_end(outputs)
        """

        print('validating...')

        # enable eval mode
        model.zero_grad()
        model.eval()

        # disable gradients to save memory
        torch.set_grad_enabled(False)

        # bookkeeping
        outputs = []

        # run training
        for i, data_batch in enumerate(dataloader):

            if data_batch is None:
                continue

            # stop short when on fast dev run
            if max_batches is not None and i >= max_batches:
                break

            # -----------------
            # RUN VALIDATION STEP
            # -----------------
            output = model.validation_step(data_batch)
            outputs.append(output)

        # give model a chance to do something with the outputs
        val_results = model.validation_end(outputs)

        # enable train mode again
        model.train()

        # enable gradients to save memory
        torch.set_grad_enabled(True)
        return val_results