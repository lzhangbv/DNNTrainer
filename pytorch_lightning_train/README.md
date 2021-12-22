# Study from Pytorch Lightning Train Design
Once you’ve organized your PyTorch code into a LightningModule, the Trainer automates everything else.

Refer to: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html and https://github.com/PyTorchLightning/pytorch-lightning/tree/master/pytorch_lightning/trainer

## Lightning Philosophy
Lightning structures your deep learning code in 4 parts: 
* Research Code
* Engineering Code
* Non-essential Code
* Data Code

### Research Code
The research code would be the particular model and how it’s trained. This code is organized into a lightning module. 

### Engineering Code
The Engineering code is all the code related to training this system, such as early stopping, distribution over GPUs, 16-bit precision, etc. This code is abstracted out by the trainer. 

### Non-essential Code
This is code that helps the research like log to tensorboard. This code is organized into callbacks. 

### Data Code
DataModules are optional but encouraged, otherwise you can use standard DataLoaders. This code is organized inside a datamodules. 