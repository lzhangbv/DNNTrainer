# Pytorch-lightning 0.0.1
The Keras for ML-researchers in PyTorch.    

## Usage
To use lightning do 2 things:  
1. Define a trainer 
2. Define a model   

## Trainer
check models/trainer.py (**fit**)

check utils/pt_callbacks.py

* early stopping: on_epoch_end (return stop_training)
* model checkpoint: on_epoch_end (save model)

## Model
check root_module/root_module.py

* data: tng_dataloader, val_dataloader, test_dataloader
* model: forward
* optimizer: configure_optimizers
* unit: training_step, validation_step, validation_end, 
* Model IO: load_model_specific, get_save_dict
* hooks (optional): on_batch_start, on_batch_end, on_epoch_start, on_epoch_end

