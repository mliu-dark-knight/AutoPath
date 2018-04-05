import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def load_node(path):
	name_to_id, id_to_name = {}, []
	type_to_node, node_to_type = defaultdict(list), []
	node_id, type_id = 0, 0
	with open(path) as f:
		for line in f:
			line = line.rstrip().split('\t')
			name_to_id[line[0]] = node_id
			id_to_name.append(line[0])
			type_to_node[line[1]].append(node_id)
			node_to_type.append(line[1])
			node_id += 1
	type_to_node_copy = {}
	for type, ids in type_to_node.items():
		type_to_node_copy[type] = np.array(ids)

	type_id = 0
	type_to_id, id_to_type = {}, []
	for id, t in enumerate(type_to_node.keys()):
		type_to_id[t] = id
		id_to_type.append(t)
		type_id += 1
	return id_to_name, name_to_id, node_to_type, type_to_node_copy, id_to_type, type_to_id


def load_pair(path):
	pairs = defaultdict(set)
	with open(path) as f:
		for line in  f:
			line = line.rstrip().split('\t')
			if len(line) == 2:
				pairs[line[0]].add(line[1])
				pairs[line[1]].add(line[0])
	return pairs


def load_groups(paths):
	groups = []
	for path in paths:
		group = set()
		with open(path) as f:
			for line in f:
				group.add(line.rstrip())
		groups.append(group)
	return groups


def precision_recall(prediction, ground_truth, top_k):
	assert len(prediction) == len(ground_truth)
	precision, recall = 0.0, 0.0
	for key, recommendation in prediction.items():
		intersection = ground_truth[key] & set(recommendation[:top_k])
		precision += len(intersection) / float(top_k)
		recall += len(intersection) / float(len(ground_truth[key]))
	return precision / len(prediction), recall / len(ground_truth)


def plot(data, plot_file):
	plt.figure()
	plt.plot(range(len(data)), data)
	plt.savefig(plot_file)
	plt.close()
