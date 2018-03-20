import numpy as np
import utils
from multiprocessing import *


class Environment(object):
	def __init__(self, params):
		self.params = params
		self.load_embed()
		self.sigma = np.std(self.embedding, axis=0)
		self.load_pair()

	def load_embed(self):
		paths = [self.params.data_dir + node_type + '.txt' for node_type in self.params.node_type]
		self.id_to_name, self.name_to_id, self.embedding = utils.load_embed(paths)
		self.params.num_node = len(self.embedding)

	def load_pair(self):
		pairs = utils.load_pair(self.params.pair_file)
		self.pairs = []
		for pair in pairs:
			self.pairs.append(np.array((self.name_to_id[pair[0]], self.name_to_id[pair[1]])))
		self.pairs = np.array(self.pairs)

	# returns a 2d array, 2nd dimension is 2
	def initial_state(self):
		return self.pairs[np.random.randint(len(self.pairs), size=self.params.batch_size)]

	def reward_multiprocessing(self, initial_states, actions):
		def worker(worker_id):
			for idx, initial_state, action in zip(range(len(initial_states)), initial_states, actions):
				if idx % num_process == worker_id:
					queue.put((action, np.array(self.trajectory_reward(initial_state, action))))

		assert len(initial_states) == len(actions)
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


	def trajectory_reward(self, state, actions):
		rewards = []
		reward = 0.0
		start, target = state
		prev = -1
		for action in actions:
			rewards.append(reward)
			if action == prev:
				reward += 1.0
			elif action == target:
				reward -= 1.0
		rewards = reward - np.array(rewards)
		return rewards
