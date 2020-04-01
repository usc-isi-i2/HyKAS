import json 
import csv
import tqdm
from collections import Counter
import pickle 

DELIMITER=' '
CSKG_EDGES_FILE='cskg/edges_v003.csv'

def extract_node_data(node_file):
	node2label={}
	node2pos={}
	with open(node_file, 'r') as f:
		next(f)
		for line in f:
			fs=line.split('\t')
			node2label[fs[0].strip()]=fs[1].strip()
			node2pos[fs[0].strip()]=fs[3].strip()
	print('node index ready')
	return node2label, node2pos

def assign_if_exists(d, entry):
	if entry in d.keys():
		return d[entry]
	else:
		return ''

def read_commonsense(fl):
	en_concepts = {}
	rel_types = {}
	long_en_concepts = {}
	word_pos = {}
	concept_len = Counter()

	nodes_path=fl.replace('edges', 'nodes')
	node2label, node2pos=extract_node_data(nodes_path)
	with open(fl, 'r') as f:
		next(f)
		reader = csv.reader(f, delimiter='\t')
		for i, line in enumerate(reader):
			rela=line[1]
			#rela = '/'.join(line[1].split('/')[2:]) 
			if rela not in rel_types:
				rel_types[rela] = 1

			start_label=assign_if_exists(node2label, line[0])
			end_label=assign_if_exists(node2label, line[2])
			if start_label =='' or end_label=='': continue

			start_sense=assign_if_exists(node2pos, line[0])
			end_sense=assign_if_exists(node2pos, line[2])
			if ',' in start_sense: start_sense=''
			if ',' in end_sense: end_sense=''

			concept_len[len(start_label.split(DELIMITER))] += 1
			concept_len[len(end_label.split(DELIMITER))] += 1

			if start_sense not in word_pos:
				word_pos[start_sense] = 1

			if end_sense not in word_pos:
				word_pos[end_sense] = 1

			#meta = json.loads(line[-1])
			concept = (start_sense, rela, end_label, end_sense, float(line[3])) # meta['weight'])
			if len(start_label.split(DELIMITER)) > 1:
				for w in start_label.split(DELIMITER):
					if w not in long_en_concepts:
						long_en_concepts[w] = {start_label: [concept]}
					elif start_label not in long_en_concepts[w]:
						long_en_concepts[w][start_label] = [concept]
					else:
						long_en_concepts[w][start_label].append(concept)
			else:
				if start_label not in en_concepts:
					en_concepts[start_label] = [concept]
				else:
					en_concepts[start_label].append(concept)
			if i % 5000000 == 0:
				print (i)
	print ('one word concepts', len(en_concepts))
	print ('long concepts', len(long_en_concepts))
	print ('pos', word_pos)
	#print (rel_types)
	#print (concept_len)
	return en_concepts, long_en_concepts

def save_dict(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    en_concepts, long_en_concepts = read_commonsense(CSKG_EDGES_FILE)
    save_dict('en_concepts.pickle', en_concepts)
    save_dict('long_en_concepts.pickle', long_en_concepts)
