import torch
import os
import re


class ModelIO(object):

    def load_model_specific(self, checkpoint):
        """
        Do something with the checkpoint
        :param checkpoint:
        :return:
        """
        raise NotImplementedError

    def get_save_dict(self):
        """
        Return specific things for the model
        :return:
        """
        raise NotImplementedError


class TrainerIO(object):

    # --------------------
    # MODEL SAVE CHECKPOINT
    # --------------------
    def save_checkpoint(self, filepath):
        checkpoint = self.dump_checkpoint()

        # do the actual save
        torch.save(checkpoint, filepath)

    def dump_checkpoint(self):
        checkpoint = {
            'epoch': self.current_epoch,
            'checkpoint_callback_best': self.checkpoint_callback.best,
            'early_stop_callback_wait': self.early_stop_callback.wait,
            'early_stop_callback_patience': self.early_stop_callback.patience,
            'global_step': self.global_step
        }

        optimizer_states = []
        for i, optimizer in enumerate(self.optimizers):
            optimizer_states.append(optimizer.state_dict())

        checkpoint['optimizer_states'] = optimizer_states

        # request what to save from the model
        checkpoint_dict = self.model.get_save_dict()

        # merge trainer and model saving items
        checkpoint.update(checkpoint_dict)
        return checkpoint


    def restore_training_state(self, checkpoint):
        """
        Restore trainer state.
        Model will get its change to update
        :param checkpoint:
        :return:
        """
        self.checkpoint_callback.best = checkpoint['checkpoint_callback_best']
        self.early_stop_callback.wait = checkpoint['early_stop_callback_wait']
        self.early_stop_callback.patience = checkpoint['early_stop_callback_patience']
        self.global_step = checkpoint['global_step']

        # restore the optimizers
        optimizer_states = checkpoint['optimizer_states']
        for optimizer, opt_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(opt_state)
