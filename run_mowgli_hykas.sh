#! /bin/bash

mkdir -p output

#rm output/*
#rm checkpoint/*.mdl

cfg="cfg/hykas.yaml"

#datasets="hellaswag-train-dev" #"alphanli" #"physicaliqa-train-dev" # "socialiqa-train-dev"
#datasets="se2018t11"
datasets="csqa"

for dataset in $datasets 
do
	python -m mowgli --dataset $dataset --output output/ --config $cfg 
done
