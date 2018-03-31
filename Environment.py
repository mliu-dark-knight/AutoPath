import numpy as np
import utils
from multiprocessing import *
from scipy.sparse import csr_matrix


class Environment(object):
	def __init__(self, params):
		self.params = params
		self.load_node()
		self.load_graph()
		self.train_data = self.load_train(self.params.train_files)
		self.test_data = self.load_test(self.params.test_file)

	def load_node(self):
		self.id_to_name, self.name_to_id, self.type_to_node, self.id_to_type, self.type_to_id = \
			utils.load_node(self.params.node_file)
		self.params.num_node = len(self.id_to_name)
		self.params.num_type = len(self.id_to_type)

	def load_graph(self):
		row, col, data = [], [], []
		with open(self.params.link_file) as f:
			for line in f:
				line = line.rstrip().split('\t')
				if len(line) == 2 and line[0] in self.name_to_id and line[1] in self.name_to_id:
					row.append(self.name_to_id[line[0]])
					col.append(self.name_to_id[line[1]])
					data.append(1.0)
		self.graph = csr_matrix((data, (row, col)), shape=(self.params.num_node, self.params.num_node))

	def load_train(self, paths):
		groups = utils.load_groups(paths)
		return [np.array([self.name_to_id[name] for name in group if name in self.name_to_id ]) for group in groups]

	def load_test(self, path):
		related = utils.load_pair(path)
		return_related = {}
		for key, values in related.items():
			if key in self.name_to_id:
				ids = set([self.name_to_id[value] for value in values if value in self.name_to_id])
				if bool(ids):
					return_related[self.name_to_id[key]] = ids
		return return_related

	# returns a batch of 2d array, 2nd dimension is 2
	def initial_state(self):
		total_size = sum([len(group)**2 for group in self.train_data])
		batch_size = self.params.batch_size
		states = []
		for group in self.train_data:
			sample_size = max(len(group)**2 * batch_size / total_size, 1)
			states.append(np.stack([np.random.choice(group, size=sample_size), np.random.choice(group, size=sample_size)], axis=1))
		return np.concatenate(states, axis=0)

	# returns all test pairs
	def initial_test(self):
		states = np.array(self.test_data.keys())
		empty_states = np.array([self.params.num_node] * len(states))
		return np.stack([states, empty_states], axis=1)

	def next_state(self, starts, actions):
		nexts = []
		for start, action in zip(starts, actions):
			if action == start[1]:
				nexts.append(start)
			else:
				nexts.append(np.array((action, start[1])))
		return np.array(nexts)

	def get_neighbors(self, nodes):
		indices = []
		for node in nodes:
			indices.append(self.graph[node].nonzero()[1])
		max_length = min(self.params.max_neighbor, max(map(lambda l: len(l), indices)))
		indices_copy = []
		for index in indices:
			if len(index) < max_length:
				indices_copy.append(np.pad(index, (0, max_length - len(index)), 'wrap'))
			else:
				indices_copy.append(np.random.choice(index, max_length, replace=False))
		return np.array(indices_copy)

	def compute_reward(self, states, actions):
		assert len(states) == len(actions)
		return_states, return_actions, return_rewards = [], [], []
		for state, action in zip(states, actions):
			return_states.append(state)
			return_actions.append(action)
			return_rewards.append(np.array(self.trajectory_reward(state, action)))
		return np.concatenate(return_states, axis=0), np.concatenate(return_actions, axis=0), np.concatenate(return_rewards, axis=0)


	def trajectory_reward(self, states, actions):
		rewards = []
		reward = 0.0
		start, target = states[0]
		prev = -1
		for action in actions:
			rewards.append(reward)
			if action == prev:
				reward -= 1.0
			elif action == target:
				reward += 1.0
			prev = action
		rewards = reward - np.array(rewards)
		return rewards
