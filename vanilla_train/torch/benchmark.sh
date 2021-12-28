#!/bin/bash

# clsuter settings
ngpu_per_node="${ngpu_per_node:-4}"
node_count="${node_count:-1}"
node_rank="${node_rank:-1}"

# imagenet settings
script=imagenet_benchmark.py
params="--model resnet50 --batch-size 32"
ngpu_per_node=$ngpu_per_node node_count=$node_count node_rank=$node_rank script=$script params=$params bash launch_torch.sh