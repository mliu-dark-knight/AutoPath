import numpy as np
from collections import defaultdict


def load_node(path):
	name_to_id, id_to_name = {}, []
	type_to_node = defaultdict(list)
	node_id, type_id = 0, 0
	with open(path) as f:
		for line in f:
			line = line.rstrip().split('\t')
			name_to_id[line[0]] = node_id
			id_to_name.append(line[0])
			type_to_node[line[1]].append(node_id)
			node_id += 1
	type_to_node_copy = {}
	for type, ids in type_to_node.items():
		type_to_node_copy[type] = np.array(ids)

	type_id = 0
	type_to_id, id_to_type = {}, []
	for id, t in enumerate(type_to_node.keys()):
		type_to_id[t] = id
		id_to_type.append(type_id)
		type_id += 1
	return id_to_name, name_to_id, type_to_node_copy, id_to_type, type_to_id


def load_pair(path):
	pairs = []
	with open(path) as f:
		for line in  f:
			line = line.rstrip().split('\t')
			if len(line) == 2:
				pairs.append((line[0], line[1]))
	return pairs
