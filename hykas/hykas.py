from mowgli.classes import Dataset, Entry
import mowgli.utils.general as utils
from mowgli.predictor.predictor import Predictor

#from trian.trian_classes import SpacyTokenizer
#from trian.preprocess_utils import preprocess_dataset, preprocess_cskg
#from trian.preprocess_utils import build_vocab
#from trian.utils import load_vocab, load_data
#from trian.model import Model
from hykas.config import AttrDict, model_args as margs, preprocessing_args as pargs
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
	def preprocess(self, dataset:Dataset) -> Any:

		pp_args=AttrDict(pargs)

		
		# Preprocess KG
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

		dev_data=getattr(dataset, 'dev')

		cs_filter=[]
		for idx, sample in tqdm.tqdm(enumerate(dev_data)):
			concept, question=sample.question
			question=question.lower()
			options_cs = build_trees(en_concepts, long_en_concepts, stopwords, question, sample.answers) 
			choice_commonsense = [[],[],[],[],[]]
			common_cs = set(options_cs[0]).intersection(set(options_cs[1])).intersection(set(options_cs[2])).intersection(set(options_cs[3])).intersection(set(options_cs[4]))
			for i, o in enumerate(options_cs):
				for c in o:
					if c not in common_cs:
						choice_commonsense[i].append(c)
			cs_filter.append({'choice_commonsense': choice_commonsense, 'id': sample.id})

		with open(pp_args.cskg_filter, 'w') as fout:
			for sample in cs_filter:
				json.dump(sample, fout)
				fout.write('\n')

		return dataset

	def train(self, train_data:List, dev_data: List, graph: Any) -> Any:

		model_args=AttrDict(margs)
		run_hykas(model_args)
		"""
		model_args=AttrDict(margs)
		pp_args=AttrDict(pargs)
		print('Model arguments:', model_args)
		if model_args.pretrained:
			assert all(os.path.exists(p) for p in model_args.pretrained.split(',')), 'Checkpoint %s does not exist.' % model_args.pretrained

		train_data = load_data(pp_args.processed_file % 'train')
		train_data += load_data(pp_args.processed_file % 'trial')
		dev_data = load_data(pp_args.processed_file % 'dev') 

		load_vocab(pp_args, train_data+dev_data)

		torch.manual_seed(model_args.seed)
		np.random.seed(model_args.seed)
		random.seed(model_args.seed)

		best_model=None

		if model_args.test_mode:
			# use validation data as training data
			train_data += dev_data
			dev_data = []
		model = Model(model_args)

		best_dev_acc = 0.0
		os.makedirs(model_args.checkpoint_dir, exist_ok=True)
		checkpoint_path = '%s/%d-%s.mdl' % (model_args.checkpoint_dir, model_args.seed, datetime.now().isoformat())
		print('Trained model will be saved to %s' % checkpoint_path)
		for i in range(model_args.epoch):
			print('Epoch %d...' % i)
			if i == 0:
				print('Dev data size', len(dev_data))
				dev_acc, dev_preds, dev_probs = model.evaluate(dev_data)
				print('Dev accuracy: %f' % dev_acc)
			start_time = time.time()
			np.random.shuffle(train_data)
			cur_train_data = train_data
		
			model.train(cur_train_data)
			train2000=train_data[:2000]
			train_acc, *rest = model.evaluate(train2000, debug=False, eval_train=True)
			print('Train accuracy: %f' % train_acc)
			dev_acc, dev_preds, dev_probs = model.evaluate(dev_data, debug=True)
			print('Dev accuracy: %f' % dev_acc)

			if dev_acc > best_dev_acc:
				best_dev_acc = dev_acc
				os.system('mv %s %s ' % (model_args.last_log, model_args.best_log))
				model.save(checkpoint_path)
				#best_model = Model(model_args)
				#best_model.network.load_state_dict(copy.deepcopy(model.network.state_dict()))
			elif model_args.test_mode:
				model.save(checkpoint_path)
			print('Epoch %d use %d seconds.' % (i, time.time() - start_time))

		print('Best dev accuracy: %f' % best_dev_acc)
		"""
		return model

	def predict(self, model: Any, dataset: Dataset, partition: str) -> List:

		pp_args=AttrDict(pargs)
		dev_data = load_data(pp_args.processed_file % partition)

		dev_acc, dev_preds, dev_probs = model.evaluate(dev_data)
		print('Predict fn: Dev accuracy: %f' % dev_acc)

		#print(dev_preds)
		#print(dev_probs)
		return dev_preds, dev_probs
