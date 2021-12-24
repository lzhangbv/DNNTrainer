# The first public release

PyTorch Lightning 0.4.0 is the first public release after a short period testing with public users. 

## New Features

* Distributed training: DP or DDP
* 16-bit training: NVIDIA apex with a single GPU

For ease of presentation, we illustrate DDP only. 

## Trainer

### models/trainer.py (fit)

* single gpu train
* ddp train

### pt_overrides/override_data_parallel.py

* DDP was used to wrap PyTorch's module model
* LightningDDP was used to wrap root_module model to support ddp forward