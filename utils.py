import numpy as np

def load_embed(paths):
	name_to_id, id_to_name = {}, []
	embedding = []
	id = 0
	for path in paths:
		with open(path) as f:
			for line in f:
				line = line.rstrip().split('\t')
				name_to_id[line[0]] = id
				id_to_name.append(line[0])
				id += 1
				embedding.append(np.array(list(map(float, line[1].split()))))
	return id_to_name, name_to_id, np.array(embedding)

def load_pair(path):
	pairs = []
	with open(path) as f:
		for line in  f:
			line = line.rstrip().split('\t')
			if len(line) == 2:
				pairs.append((line[0], line[1]))
	return pairs
