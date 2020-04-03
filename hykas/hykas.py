from mowgli.classes import Dataset, Entry
import mowgli.utils.general as utils
from mowgli.predictor.predictor import Predictor

from hykas.config import get_pp_args, get_model_args, kg_name
from hykas.extract_cskg import read_commonsense
import hykas.utils
from hykas.preprocess import build_dict, build_trees
from hykas.run import run_hykas

import copy
import json
from typing import List, Any
import os
import time
import torch
import random
import numpy as np
from datetime import datetime
import tqdm
import pickle

class Hykas(Predictor):
	def preprocess(self, dataset:Dataset, config:Any) -> Any:


		kg=config['kg']
		dataname=config['dataset']
		pp_args=get_pp_args(dataname, kg)
		
		# Preprocess KG
		# the resulting files are indexed on the label of the edge subject
		# so the keys are these labels, and the values are lists of tuples that include the predicate and the object
		if os.path.exists(pp_args.short_concepts_pkl):
			en_concepts=pickle.load(open(pp_args.short_concepts_pkl, 'rb'))
			long_en_concepts=pickle.load(open(pp_args.long_concepts_pkl, 'rb'))
			print(len(en_concepts), 'concepts, ', len(long_en_concepts), 'long concepts.')
		else:
			en_concepts, long_en_concepts = read_commonsense(pp_args.kg_edges)
			hykas.utils.save_dict(pp_args.short_concepts_pkl, en_concepts)
			hykas.utils.save_dict(pp_args.long_concepts_pkl, long_en_concepts)

		# Preprocess dataset
		train_data = getattr(dataset, 'train')
		vocab, stopwords = build_dict(train_data)

		print('vocab size:', len(vocab))
		print(vocab)
		print('stopwords:', len(stopwords))
		print(stopwords)


		# Lookup the vocabulary in the CSKG to create relevant background knowledge
		for partition in pp_args.partitions:
			part_data=getattr(dataset, partition)
			cs_filter=[]
			for idx, sample in tqdm.tqdm(enumerate(part_data)):
				concept, question=sample.question
				question=question.lower()
				options_cs = build_trees(en_concepts, long_en_concepts, stopwords, question, sample.answers) 
				choice_commonsense = [[],[],[],[],[]]
				common_cs = set(options_cs[0]).intersection(*options_cs)
				for i, o in enumerate(options_cs):
					for c in o:
						if c not in common_cs:
							choice_commonsense[i].append(c)
				cs_filter.append({'choice_commonsense': choice_commonsense, 'id': sample.id})
			hykas.utils.save_jsonl(pp_args.cskg_filter[partition], cs_filter)
		return dataset

	def train(self, dataset:Dataset, config:Any) -> Any:
		kg=config['kg']     
		dataname=config['dataset']
		model_args=get_model_args(dataname, kg)
		pp_args=get_pp_args(dataname, kg)

		commonsense={}
		for part in pp_args.partitions:
			with open(pp_args.cskg_filter[part], 'r') as f:
				commonsense[part]=f.readlines()
			
		train_data=getattr(dataset, 'train')
		dev_data=getattr(dataset, 'dev')

		model, results=run_hykas(model_args, train_data, dev_data, commonsense)
		return model

	def predict(self, model: Any, dataset: Dataset, config:Any, partition: str) -> List:
		kg=config['kg']
		dataname=config['dataset']
		pp_args=get_pp_args(dataname, kg)
		exit(0)
#		dev_acc, dev_preds, dev_probs = model.evaluate(dev_data)
#		print('Predict fn: Dev accuracy: %f' % dev_acc)

#		return dev_preds, dev_probs
