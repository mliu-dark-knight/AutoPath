import numpy as np
import utils
from multiprocessing import *
from scipy.sparse import csr_matrix


class Environment(object):
	def __init__(self, params):
		self.params = params
		self.load_node()
		self.load_graph()
		self.train_pairs = self.load_pair(self.params.train_file)
		self.test_pairs = self.load_pair(self.params.test_file)

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

	def load_pair(self, path):
		pairs = utils.load_pair(path)
		return_pairs = []
		for pair in pairs:
			if pair[0] in self.name_to_id and pair[1] in self.name_to_id:
				return_pairs.append(np.array((self.name_to_id[pair[0]], self.name_to_id[pair[1]])))
		return np.array(return_pairs)

	# returns a 2d array, 2nd dimension is 2
	def initial_state(self):
		return self.train_pairs[np.random.randint(len(self.train_pairs), size=self.params.batch_size)]

	def initial_test(self):
		return self.test_pairs

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

	def reward_multiprocessing(self, states, actions):
		def worker(worker_id):
			for idx, state, action in zip(range(len(states)), states, actions):
				if idx % num_process == worker_id:
					queue.put((state, action, np.array(self.trajectory_reward(state, action))))

		assert len(states) == len(actions)
		num_process = self.params.num_process
		queue = Queue()
		processes = []
		for id in range(num_process):
			process = Process(target=worker, args=(id,))
			process.start()
			processes.append(process)

		ret_states, ret_actions, ret_rewards = [], [], []
		for i in range(self.params.batch_size):
			state, action, reward = queue.get()
			ret_states.append(state)
			ret_actions.append(action)
			ret_rewards.append(reward)

		for process in processes:
			process.join()

		return np.concatenate(ret_states, axis=0), np.concatenate(ret_actions, axis=0), np.concatenate(ret_rewards, axis=0)

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
