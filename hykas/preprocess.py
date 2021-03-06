import json 
import csv
import tqdm
from collections import Counter
import operator
import pickle
from os.path import isfile
from nltk import ngrams
from hykas import tokenization
tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
import nltk
import operator
allowed_pos = ['JJ', 'NN', 'VB', 'RB']
delimiter=' '
min_overlap=0.5
num_stopwords=20

class Commonsense_GV(object):
	def __init__(self, start, end):
		self.start = start
		self.end = end
		self.relation = {}
		
	def add_relation(self, full_start, full_end, r):
		k = (full_start, r[1], full_end)
		self.relation[k] = r[-1]
		
	def add_long_relation(self, full_start, r1, r2):
		k = (full_start, r1[1], r1[2], r2[1], r2[2])
		self.relation[k] = max(r1[-1], r2[-1])
		
	def get_relation(self):
		#print (self.relation)
		return max(self.relation.items(), key=operator.itemgetter(1))[0]

def get_ngrams(tokens):
	curr = len(tokens)-1
	all_grams = []
	while curr > 1:
		for gram in ngrams(tokens, curr):
			all_grams.append(' '.join(gram))
		curr -= 1
	return all_grams

def addedge(graph, start, end, full_start, full_end, rel):
	#if full_start == rel[2]:
	#    return
	found = 0
	for edge in graph:
		if edge.start == start and edge.end == end:
			edge.add_relation(full_start, full_end, rel)
			found = 1
			break
	if found == 0:
		graph.append(Commonsense_GV(start, end))
		graph[-1].add_relation(full_start, full_end, rel)
	   
def build_dict(data):
	vocab = Counter()
	for sample in data:
		context, question=sample.question
		for w in tokenizer.tokenize(question): 
			vocab[w] += 1
		for choice in sample.answers:
			for w in tokenizer.tokenize(choice):
				vocab[w] += 1
	print (len(vocab))
	stopwords = dict(vocab.most_common(num_stopwords))
	print (stopwords)
	return vocab, stopwords
				
def check_list(word, l, pos=None):                     # this checks that word appear in context and its pos is in allowed list, if target concepts has pos, it needs to match the words' pos
	for w in l:
		if w[0] == word and w[1][:2] in allowed_pos:
			if pos == None or match_pos(w, pos):
				return True
	return False

def match_pos(word, pos):
	if pos == '' or len(word[0].split(delimiter)) > 1:
		return True
	if word[1][:2] == 'NN' and pos == 'n':
		return True
	if word[1][:2] == 'VB' and pos == 'v':
		return True
	if word[1][:2] == 'JJ' and pos == 'a':
		return True
	if word[1][:2] == 'RB' and pos == 'a':
		return True
	return False

def check_valid(subj_anchor, subj, obj, obj_pos, target_seq, target_string, stopwords):
	if obj in stopwords or obj == subj or obj == subj_anchor or len(obj) < 3:
		return False, None
	if delimiter in obj:
		if obj in target_string:
			return True, obj
		else:
			obj_tokens = obj.split(delimiter)
			matchs, target_word = get_matching_words(obj_tokens, target_seq, target_string, stopwords, subj, subj_anchor)  
			if float(matchs)/len(obj_tokens) > min_overlap:
				return True, target_word
	elif check_list(obj, target_seq, pos=obj_pos):
		return True, obj
	
	return False, None
	
def get_matching_words(obj_tokens, target_seq, target_string, stopwords, subj, subj_anchor):   # just pick the first word for now 
	obj_ngrams = get_ngrams(obj_tokens)
	for gram in obj_ngrams:
		if gram in target_string:
			return len(gram.split(delimiter)), gram
	count = 0
	match = None
	subj_tokens = subj.split(delimiter)
	for w in obj_tokens:
		if w not in stopwords and w != subj and w != subj_anchor and w not in subj_tokens and check_list(w, target_seq):
			count += 1
			match = w
	return count, match


def build_trees(en_concepts, long_en_concepts, stopwords, question, options):
	#print (question, answer)
	question_tokens = tokenizer.tokenize(question)
	question_pos = nltk.pos_tag(question_tokens)
	question_pos = list(set(question_pos))
	options_cs = []
	for op in options:
		graph = []
		option_tokens = tokenizer.tokenize(op)
		option_pos = nltk.pos_tag(option_tokens)
		option_pos = list(set(option_pos))
		for word in question_pos:
			if word[0] not in stopwords and word[1][:2] in allowed_pos and word[0] in en_concepts:
				for rel in en_concepts[word[0]]:
					in_context, match_obj = check_valid(word[0], word[0], rel[2], rel[3], option_pos, op, stopwords)
					if in_context:
						addedge(graph, word[0], match_obj, word[0], rel[2], rel)
			if word[0] not in stopwords and word[1][:2] in allowed_pos and word[0] in long_en_concepts:
				for phrase in long_en_concepts[word[0]]:
					in_origin, match_subj = check_valid('-', '-', phrase, '', question_pos, question, stopwords)
					if in_origin:
						for rel in long_en_concepts[word[0]][phrase]:
							in_context, match_obj = check_valid(word[0], phrase, rel[2], rel[3], option_pos, op, stopwords)
							if in_context:
								addedge(graph, word[0], match_obj, phrase, rel[2], rel)
		for word in option_pos:
			if word[0] not in stopwords and word[1][:2] in allowed_pos and word[0] in en_concepts:
				for rel in en_concepts[word[0]]:
					in_context, match_obj = check_valid(word[0], word[0], rel[2], rel[3], question_pos, question, stopwords)
					if in_context:
						addedge(graph, word[0], match_obj, word[0], rel[2], rel)
			if word[0] not in stopwords and word in long_en_concepts:
				for phrase in long_en_concepts[word]:
					in_origin, match_subj = check_valid('-', '-', phrase, '', option_pos, op, stopwords)
					if in_origin:
						for rel in long_en_concepts[word][phrase]:
							in_context, match_obj = check_valid(word, phrase, rel[2], rel[3], question_pos, question, stopwords)
							if in_context:
								addedge(graph, word, match_obj, phrase, rel[2], rel)
		options_cs.append(list(set([v.get_relation() for v in graph])))
		#print ()
	return options_cs
