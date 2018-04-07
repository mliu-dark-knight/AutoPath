import numpy as np
from scipy.sparse import csr_matrix

import utils


class Environment(object):
	def __init__(self, params):
		self.params = params
		self.load_node()
		self.load_graph()
		self.train_data = self.load_train(self.params.train_files)
		self.test_pos = self.load_test(self.params.test_pos_file)
		self.test_neg = self.load_test(self.params.test_neg_file)


	def load_node(self):
		id_to_name, self.name_to_id, node_to_type, self.type_to_node, id_to_type, self.type_to_id = \
			utils.load_node(self.params.node_file)
		self.id_to_name = np.array(id_to_name)
		self.node_to_type = np.array(node_to_type)
		self.id_to_type = np.array(id_to_type)
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
		return_groups = [np.array([self.name_to_id[name] for name in group if name in self.name_to_id])
		                 for group in groups]
		print('Train size: %d' % sum([len(group)**2 for group in return_groups]))
		return return_groups


	def load_test(self, path):
		related = utils.load_pair(path)
		return_related = {}
		for key, values in related.items():
			if key in self.name_to_id:
				ids = set([self.name_to_id[value] for value in values if value in self.name_to_id])
				if bool(ids):
					return_related[self.name_to_id[key]] = ids
		print('Test size: %d' % sum(map(len,  return_related.values())))
		return return_related


	# returns a batch of 2d array, 2nd dimension is 2
	def initial_state(self):
		total_size = sum([len(group)**2 for group in self.train_data])
		batch_size = self.params.batch_size
		states = []
		for group in self.train_data:
			sample_size = max(len(group)**2 * batch_size / total_size, 1)
			state = np.random.choice(group, size=sample_size)
			states.append(np.stack([state, state], axis=1))
		return np.concatenate(states, axis=0)


	# returns all test nodes
	def initial_test(self):
		states = np.array(self.test_pos.keys())
		return np.stack([states, states], axis=1)


	def get_neighbors(self, nodes):
		indices = []
		for node in nodes:
			indices.append(self.graph[node].nonzero()[1])
		# max_length = 0 will cause error
		max_length = max(min(self.params.max_neighbor, max(map(lambda l: len(l), indices))), 1)
		indices_copy = []
		for index in indices:
			# select random nodes as neighbors
			if len(index) == 0:
				indices_copy.append(np.random.choice(self.params.num_node, size=max_length))
			elif len(index) < max_length:
				indices_copy.append(np.pad(index, (0, max_length - len(index)), 'wrap'))
			else:
				indices_copy.append(np.random.choice(index, max_length, replace=False))
		return np.array(indices_copy)


	def compute_reward(self, states, actions, nexts):
		assert len(states) == len(actions) and len(actions) == len(nexts) and len(actions) == len(nexts)
		return_states, return_actions, return_rewards, return_nexts, return_future_rewards = \
			[], [], [], [], []
		for state, action, next in zip(states, actions, nexts):
			reward, future_reward = self.trajectory_reward(state, action)
			return_states.append(state)
			return_actions.append(action)
			return_rewards.append(np.array(reward))
			return_nexts.append(next)
			return_future_rewards.append(np.array(future_reward))
		return np.concatenate(return_states, axis=0), \
		       np.concatenate(return_actions, axis=0), \
		       np.concatenate(return_rewards, axis=0), \
		       np.concatenate(return_nexts, axis=0), \
		       np.concatenate(return_future_rewards, axis=0)


	def trajectory_reward(self, states, actions):
		rewards, future_rewards = [], []
		total_reward = 0
		start = states[0][0]
		start_group = set()
		for i, group in enumerate(self.train_data):
			if start in group:
				start_group.add(i)
		assert bool(start_group)
		for action in actions:
			future_rewards.append(total_reward)
			reward = 0
			if action < self.params.num_node and self.node_to_type[action] == self.node_to_type[start] \
					and action != start:
				action_group = set()
				for i, group in enumerate(self.train_data):
					if action in group:
						action_group.add(i)
				intersection = start_group & action_group
				if bool(intersection):
					reward = 1
				else:
					reward = -1
				total_reward += reward
			rewards.append(reward)
		future_rewards = total_reward - np.array(future_rewards)
		return rewards, future_rewards
