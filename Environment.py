import numpy as np
import util
from multiprocessing import *
from Cube import Cube


class Environment(object):
	def __init__(self, params):
		self.params = params
		self.cube = Cube.load_cube(self.params.cube_file)
		self.cell_embed = util.load_embed(params, self.cube)
		self.sigma = np.std(self.cell_embed, axis=0)
		self.init_state = self.initial_state()

	def initial_state(self):
		return self.cube.initial_state(self.params.test_file, self.params.low_limit, self.params.high_limit, self.params.debug)

	def state_embed(self, state):
		return np.mean(self.cell_embed[state], axis=0)

	def total_reward(self, state):
		return self.cube.total_reward(state, self.params)

	def reward_multiprocessing(self, state_embeds, initial_states, actions):
		def worker(worker_id):
			for idx, state_embed, initial_state, action in zip(range(len(state_embeds)), state_embeds, initial_states, actions):
				if idx % num_process == worker_id:
					queue.put((state_embed, action, np.array(self.trajectory_reward(initial_state, action))))

		assert len(state_embeds) == len(initial_states) and len(initial_states) == len(actions)
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
		return self.cube.trajectory_reward(state, actions, self.params)

	def convert_state(self, state):
		return self.cube.all_authors(state)
