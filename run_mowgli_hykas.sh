#! /bin/bash

mkdir -p output

cfg="cfg/hykas.yaml"
python -m mowgli --config $cfg
