
kg='cskg'
dataset='csqa'

model_args={
			'data_dir': './data/%s' % kg,
			'model_type': 'roberta-ocn-inj',
			'model_name_or_path': 'roberta-large',
			'task_name': '%s-inj' % dataset,
			'overwrite_output_dir': True,
			'output_dir': './output/%s-%s' % (kg, dataset),
			'config_name': '',
			'tokenizer_name': '',
			'cache_dir': 'downloaded_models',
			'max_seq_length': 80,
			'do_train': True,
			'do_eval': True,
			'test': False,
			'evaluate_during_training': True,
			'do_lower_case': False,
			'per_gpu_train_batch_size': 4,
			'per_gpu_eval_batch_size': 8,
			'gradient_accumulation_steps': 8,
			'learning_rate': 1e-5,
			'weight_decay': 0.01,
			'adam_epsilon': 1e-6,
			'max_grad_norm': 1.0,
			'num_train_epochs': 8,
			'max_steps': -1,
			'warmup_steps': 150,
			'warmup_proportion': 0.1,
			'max_concepts': 12,
			'max_concept_len': 12,
			'logging_steps': 50,
			'save_steps': 50,
			'eval_all_checkpoints': False,
			'no_cuda': True,
			'overwrite_cache': False,
			'seed': 2555,
			'split_model_at': -1,
			'finetune_from': None,
			'fp16': False,
			'fp16_opt_level': '01',
			'local_rank': -1,
			'server_ip': '',
			'server_port': ''
			}


preprocessing_args={
			'kg_edges': './data/%s/edges_v004.csv' % kg,
			'short_concepts_pkl': './output/%s-%s/en_concepts.pickle' % (kg, dataset),
            'long_concepts_pkl': './output/%s-%s/long_en_concepts.pickle' % (kg, dataset),
			'cskg_filter': './output/%s-%s/cskg.filter' % (kg, dataset),
			'partitions': ['train', 'dev']
			}

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
