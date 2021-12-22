#!/bin/bash

# clsuter settings
nworkers="${nworkers:-4}"
rdma="${rdma:-1}"

# imagenet settings
script=imagenet_benchmark.py
params="--model resnet50 --batch-size 32"
nworkers=$nworkers rdma=$rdma script=$script params=$params bash launch_horovod.sh