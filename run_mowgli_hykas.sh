#! /bin/bash

mkdir -p output

cfg="cfg/hykas.yaml"
CUDA_VISIBLE_DEVICES=1,2,3 python -m mowgli --config $cfg
