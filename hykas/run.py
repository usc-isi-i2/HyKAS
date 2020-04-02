# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
								  BertTokenizer,
								  RobertaConfig,
								  RobertaTokenizer,
								  AlbertConfig, AlbertTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup

from hykas.utils import myprocessors, accuracy, output_modes, convert_examples_to_features
from hykas.models import ModelForMCRC, OCNModel

import json

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, AlbertConfig)), ())

MODEL_CLASSES = {
	'bert': (BertConfig, ModelForMCRC, BertTokenizer),
	'roberta': (RobertaConfig, ModelForMCRC, RobertaTokenizer),
	'bert-ocn': (BertConfig, OCNModel, BertTokenizer),
	'bert-ocn-inj': (BertConfig, OCNModel, BertTokenizer),
	'roberta-ocn': (RobertaConfig, OCNModel, RobertaTokenizer),
	'roberta-ocn-inj': (RobertaConfig, OCNModel, RobertaTokenizer),
	'albert': (AlbertConfig, ModelForMCRC, AlbertTokenizer),
	'albert-ocn': (AlbertConfig, OCNModel, AlbertTokenizer),
	'albert-ocn-inj': (AlbertConfig, OCNModel, AlbertTokenizer),
}


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args, train_dataset, model, tokenizer, train_data, dev_data, cs_data):
	""" Train the model """
	if args.local_rank in [-1, 0]:
		tb_writer = SummaryWriter(os.path.join('runs', args.output_dir.split('/')[-1]))

	if args.split_model_at == -1:
		args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	else:
		args.train_batch_size = args.per_gpu_train_batch_size
	train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=mCollateFn)

	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
	else:
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]

	warmup_steps = args.warmup_steps if args.warmup_steps != 0 else int(args.warmup_proportion * t_total)
	logger.info("warm up steps = %d", warmup_steps)
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(0.9, 0.98))
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

	if args.fp16:
		try:
			from apex import amp
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
		model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

	# multi-gpu training (should be after apex fp16 initialization)
	if args.n_gpu > 1 and args.split_model_at == -1:
		print('parallelizing with', args.n_gpu)
		model = torch.nn.DataParallel(model)

	# Distributed training (should be after apex fp16 initialization)
	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
														  output_device=args.local_rank,
														  find_unused_parameters=True)

	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
				   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 0
	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
	set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
	curr_best = 0.0
	for _ in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
		preds = None
		out_label_ids = None
		sample_ids = None
		for step, batch in enumerate(epoch_iterator):
			model.train()
			inputs = {'input_ids':      batch[1],
					  'attention_mask': batch[2],
					  'token_type_ids': batch[3],  # XLM and RoBERTa don't use segment_ids
					  'labels':         batch[4],
					  'concepts': batch[5],
					  'concepts_mask': batch[6],
					  'concepts_mask_full': batch[7]}
			outputs = model(**inputs)
			loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

			logits = outputs[1]
			if preds is None:
				preds = logits.detach().cpu().numpy()
				out_label_ids = inputs['labels'].detach().cpu().numpy()
				sample_ids = batch[0]
			else:
				preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
				out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
				sample_ids += batch[0]

			if args.n_gpu > 1 and args.split_model_at == -1:
				loss = loss.mean() # mean() to average on multi-gpu parallel training
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps

			if args.fp16:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
				#torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
			else:
				loss.backward()
				#torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

			tr_loss += loss.item()
			if (step + 1) % args.gradient_accumulation_steps == 0:
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()
				global_step += 1

				if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
					# Log metrics
					tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
					tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
					tb_writer.add_scalar('Batch_loss', loss.item()*args.gradient_accumulation_steps, global_step)
					logging_loss = tr_loss

			if args.max_steps > 0 and global_step > args.max_steps:
				epoch_iterator.close()
				break
		if args.max_steps > 0 and global_step > args.max_steps:
			train_iterator.close()
			break
		preds = np.argmax(preds, axis=1)
		tr_acc = accuracy(preds, out_label_ids)
		logger.info("Training acc = %s", str(tr_acc['acc']))
		tb_writer.add_scalar('train_acc', tr_acc['acc'], global_step)
		if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
			results = evaluate(args, model, tokenizer, train_data, dev_data, cs_data)
			for key, value in results.items():
				tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
			if results['acc'] > curr_best:
				curr_best = results['acc']
				# Save model checkpoint
				output_dir = args.output_dir
				if not os.path.exists(output_dir):
					os.makedirs(output_dir)
				model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
				model_to_save.save_pretrained(output_dir)
				tokenizer.save_pretrained(output_dir)
				torch.save(args, os.path.join(output_dir, 'training_args.bin'))
				logger.info("Saving model checkpoint to %s", output_dir)
	if args.local_rank in [-1, 0]:
		tb_writer.close()
	return global_step, tr_loss / global_step

def save_logits(logits_all, filename):
	with open(filename, "w") as f:
		for i in range(len(logits_all)):
			for j in range(len(logits_all[i])):
				f.write(str(logits_all[i][j]))
				if j == len(logits_all[i])-1:
					f.write("\n")
				else:
					f.write(" ")


def evaluate(args, model, tokenizer, train_data, dev_data, cs_data, prefix=""):
	# Loop to handle MNLI double evaluation (matched, mis-matched)
	eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
	eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

	results = {}
	for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
		eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, train_data, dev_data, cs_data, evaluate=True)

		if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
			os.makedirs(eval_output_dir)
		if args.split_model_at == -1:
			args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
		else:
			args.eval_batch_size = args.per_gpu_eval_batch_size
		# Note that DistributedSampler samples randomly
		eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
		eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=mCollateFn)

		# Eval!
		logger.info("***** Running evaluation {} *****".format(prefix))
		logger.info("  Num examples = %d", len(eval_dataset))
		logger.info("  Batch size = %d", args.eval_batch_size)
		eval_loss = 0.0
		nb_eval_steps = 0
		preds = None
		out_label_ids = None
		for batch in tqdm(eval_dataloader, desc="Evaluating"):
			model.eval()
			with torch.no_grad():
				inputs = {'input_ids':      batch[1],
						  'attention_mask': batch[2],
						  'token_type_ids': batch[3],  # XLM and RoBERTa don't use segment_ids
						  'labels':         batch[4],
						  'concepts': batch[5],
					  	  'concepts_mask': batch[6],
					  	  'concepts_mask_full': batch[7]}
				outputs = model(**inputs)
				tmp_eval_loss, logits = outputs[:2]

				eval_loss += tmp_eval_loss.mean().item()
			nb_eval_steps += 1
			if preds is None:
				preds = logits.detach().cpu().numpy()
				out_label_ids = inputs['labels'].detach().cpu().numpy()
			else:
				preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
				out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
		save_logits(preds, os.path.join(eval_output_dir, "logits_test.txt"))
		eval_loss = eval_loss / nb_eval_steps
		if args.output_mode == "classification":
			preds = np.argmax(preds, axis=1)
		elif args.output_mode == "regression":
			preds = np.squeeze(preds)
		result = accuracy(preds, out_label_ids)
		results.update(result)
		output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
		with open(output_eval_file, "w") as writer:
			logger.info("***** Eval results {} *****".format(prefix))
			for key in sorted(result.keys()):
				logger.info("  %s = %s", key, str(result[key]))
				writer.write("%s = %s\n" % (key, str(result[key])))

	return results

def predict(args, model, tokenizer, prefix=""):
	# Loop to handle MNLI double evaluation (matched, mis-matched)
	eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
	eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

	results = {}
	for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
		eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, test=True)

		if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
			os.makedirs(eval_output_dir)
		if args.split_model_at == -1:
			args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
		else:
			args.eval_batch_size = args.per_gpu_eval_batch_size
		# Note that DistributedSampler samples randomly
		eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
		eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=mCollateFn)

		# Eval!
		logger.info("***** Running evaluation {} *****".format(prefix))
		logger.info("  Num examples = %d", len(eval_dataset))
		logger.info("  Batch size = %d", args.eval_batch_size)
		eval_loss = 0.0
		nb_eval_steps = 0
		preds = None
		out_label_ids = None
		for batch in tqdm(eval_dataloader, desc="Evaluating"):
			model.eval()
			#batch = tuple(t.to(args.device) for t in batch)

			with torch.no_grad():
				inputs = {'input_ids':      batch[1],
						  'attention_mask': batch[2],
						  'token_type_ids': batch[3],  # XLM and RoBERTa don't use segment_ids
						  'labels':         batch[4],
						  'concepts': batch[5],
					  	  'concepts_mask': batch[6],
					  	  'concepts_mask_full': batch[7]}
				outputs = model(**inputs)
				logits = outputs[0]

			nb_eval_steps += 1
			if preds is None:
				preds = logits.detach().cpu().numpy()
			else:
				preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
		save_logits(preds, os.path.join(eval_output_dir, "logits_submission.txt"))
	return results

class MyDataset(torch.utils.data.Dataset):

	def __init__(self, data, pad_on_left, pad_token, pad_seg_id, cls_token, sep_token, num_cand):
		self.data = data
		self.pad_on_left = pad_on_left
		self.pad_token = pad_token
		self.pad_seg_id = pad_seg_id
		self.cls_token = cls_token
		self.sep_token = sep_token
		self.num_cand = num_cand

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		return sample, self.pad_on_left, self.pad_token, self.pad_seg_id, self.cls_token, self.sep_token, self.num_cand

def mCollateFn(batch):
	batch_sample_ids = []
	batch_input_ids = []
	batch_input_mask = []
	batch_token_type_ids = []
	batch_label_ids = []
	batch_commonsense_ids = []
	batch_commonsense_mask = []
	batch_commonsense_mask_full = []
	pad_on_left, pad_token, pad_seg_id, cls_token, sep_token, num_cand = batch[0][1:]
	device=torch.device('cpu')
	if pad_on_left:
		print ('this needs pad on left')
		exit(0)
	features = [b[0] for b in batch]
	for f in features:
		batch_sample_ids.append(f[0].guid)
		batch_input_ids.append([])
		batch_input_mask.append([])
		batch_token_type_ids.append([])
		batch_label_ids.append(f[0].label_id)
		if features[0][0].concepts != None:
			batch_commonsense_ids.append([])
			batch_commonsense_mask.append([])
			batch_commonsense_mask_full.append([])
		for i in range(num_cand):
			batch_input_ids[-1].append(f[i].input_ids)
			batch_input_mask[-1].append(f[i].input_mask)
			batch_token_type_ids[-1].append(f[i].token_type_ids)
			if features[0][0].concepts != None:
				batch_commonsense_ids[-1].append(f[i].concepts)
				batch_commonsense_mask_full[-1].append(f[i].concepts_mask_full)
				batch_commonsense_mask[-1].append(f[i].concepts_mask)
	if len(batch_commonsense_ids) == 0:
		batch_commonsense_ids, batch_commonsense_mask_full, batch_commonsense_mask = None, None, None
	else:
		batch_commonsense_ids = torch.tensor(batch_commonsense_ids, dtype=torch.long).to(device)
		batch_commonsense_mask_full = torch.tensor(batch_commonsense_mask_full, dtype=torch.long).to(device)
		batch_commonsense_mask = torch.tensor(batch_commonsense_mask, dtype=torch.long).to(device)
	batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(device)
	batch_input_mask = torch.tensor(batch_input_mask, dtype=torch.long).to(device)
	batch_token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long).to(device)
	if batch_label_ids[0] != None:
		batch_label_ids = torch.tensor(batch_label_ids, dtype=torch.long).to(device)
	else:
		batch_label_ids = None

	return batch_sample_ids, batch_input_ids, batch_input_mask, batch_token_type_ids, batch_label_ids, batch_commonsense_ids, batch_commonsense_mask, batch_commonsense_mask_full

def load_and_cache_examples(args, task, tokenizer, train_data, dev_data, cs_data, evaluate=False, test=False):
	if args.local_rank not in [-1, 0] and not evaluate:
		torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

	processor = myprocessors[task](train_data, dev_data, cs_data)
	output_mode = output_modes[task]
	label_list = processor.get_labels()
	if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
		# HACK(label indices are swapped in RoBERTa pretrained model)
		label_list[1], label_list[2] = label_list[2], label_list[1] 
	if test:
		examples = processor.get_test_examples() 
	else:
		examples = processor.get_dev_examples() if evaluate else processor.get_train_examples()
	pad_on_left = bool(args.model_type in ['xlnet'])
	pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
	pad_token_segment_id = 4 if args.model_type in ['xlnet'] else 0
	cls_token = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])[0]
	sep_token = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])[0]
	features = convert_examples_to_features(examples, tokenizer, label_list=label_list, max_length=args.max_seq_length,
		output_mode=output_mode,
		pad_on_left=pad_on_left,                 # pad on the left for xlnet
		pad_token=pad_token,
		pad_token_segment_id=pad_token_segment_id,
		max_path=args.max_concepts, max_path_len=args.max_concept_len, cls_token=cls_token, sep_token=sep_token
	)

	if args.local_rank == 0 and not evaluate:
		torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

	return MyDataset(features, pad_on_left, pad_token, pad_token_segment_id, cls_token, sep_token, len(label_list))

def run_hykas(args, train_data, dev_data, cs_data):

	if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
		raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	# Setup distant debugging if needed
	if args.server_ip and args.server_port:
		# Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
		import ptvsd
		print("Waiting for debugger attach")
		ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
		ptvsd.wait_for_attach()

	# Setup CUDA, GPU & distributed training
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		args.n_gpu = torch.cuda.device_count()
	else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		torch.distributed.init_process_group(backend='nccl')
		args.n_gpu = 1
	args.device = device

	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)
	# Setup logging
	log_file = os.path.join(args.output_dir, 'train.log')
	logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt = '%m/%d/%Y %H:%M:%S',
						level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
						filename=log_file)
	logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
					args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

	# Set seed
	set_seed(args)
	# Prepare GLUE task
	args.task_name = args.task_name.lower()
	if args.task_name not in myprocessors:
		raise ValueError("Task not found: %s" % (args.task_name))
	processor = myprocessors[args.task_name](train_data, dev_data, cs_data)
	args.output_mode = output_modes[args.task_name]
	label_list = processor.get_labels()
	num_labels = len(label_list)
	# Load pretrained model and tokenizer
	if args.local_rank not in [-1, 0]:
		torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

	args.model_type = args.model_type.lower()
	config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
	config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name, cache_dir=args.cache_dir)
	tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)
	model = model_class(config, args.model_type.split('-'))
	model.core = model.core.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, cache_dir=args.cache_dir)
	if 'roberta' in args.model_name_or_path:
		model.core._resize_type_embeddings(2)
	if args.finetune_from != None:
		state_dict = torch.load(os.path.join(args.finetune_from, 'pytorch_model.bin'), map_location='cpu')
		model.load_state_dict(state_dict)

	count = count_parameters(model)
	print (count)

	if args.local_rank == 0:
		torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

	model.to(args.device)

	logger.info("Training/evaluation parameters %s", args)
	if args.split_model_at != -1:
		model.redistribute(args.split_model_at)

	# Training
	if args.do_train:
		train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, train_data, dev_data, cs_data, evaluate=False)
		global_step, tr_loss = train(args, train_dataset, model, tokenizer, train_data, dev_data, cs_data)
		logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


	# Evaluation
	results = {}
	if args.do_eval and args.local_rank in [-1, 0]:
		tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
		checkpoints = [args.output_dir]
		if args.eval_all_checkpoints:
			checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
			logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
		logger.info("Evaluate the following checkpoints: %s", checkpoints)
		for checkpoint in checkpoints:
			global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
			state_dict = torch.load(os.path.join(checkpoint, 'pytorch_model.bin'), map_location='cpu')
			model.load_state_dict(state_dict)
			model.eval()
			#model = model_class.from_pretrained(checkpoint)
			model.to(args.device)
			if args.split_model_at != -1:
				model.redistribute(args.split_model_at)
			if args.test:
				result = predict(args, model, tokenizer, prefix=global_step)
			else:
				result = evaluate(args, model, tokenizer, train_data, dev_data, cs_data, prefix=global_step)
			result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
			results.update(result)

	return model, results
