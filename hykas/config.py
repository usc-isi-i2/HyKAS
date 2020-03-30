
model_args={
			'gpu': '0',
			'epoch': 50,
			'optimizer': 'adamax',
			'use_cuda': True,
			'grad_clipping': 10.0,
			'lr': 2e-3,
			'batch_size': 32,
			'embedding_file': './data/glove.840B.300d.txt',
			'hidden_size': 96,
			'doc_layers': 1,
			'rnn_type': 'lstm',
			'dropout_rnn_output': 0.4,
			'rnn_padding': True,
			'dropout_emb': 0.4,
			'pretrained': '',
			'finetune_topk': 10,
			'pos_emb_dim': 12,
			'ner_emb_dim': 8,
			'rel_emb_dim': 10,
			'seed': 1234,
			'test_mode': False,
			'checkpoint_dir': './checkpoint',
			'last_log': './output/output.log',
			'best_log': './output/best-dev.log',
			'save_loc': './output/model.mdl'
			}

preprocessing_args={
			'kg_edges': './data/cskg/edges_v004.csv',
			'short_concepts_pkl': './output/en_concepts.pickle',
            'long_concepts_pkl': './output/long_en_concepts.pickle',
			'cskg_filter': './output/cskg.filter',
			'partitions': ['train', 'dev']
			}

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
