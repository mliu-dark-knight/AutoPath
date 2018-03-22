import gc
import numpy as np
from copy import deepcopy
from NN import *
from tqdm import tqdm
from random import shuffle


class PPO(object):
	def __init__(self, params, environment):
		self.params = params
		self.environment = environment
		self.build()

	def build(self):
		self.embedding = tf.Variable(self.environment.embedding, dtype=tf.float32, trainable=True)
		self.neighbors = tf.placeholder(tf.int32, [None, None])
		self.state = tf.placeholder(tf.int32, [None])
		self.target = tf.placeholder(tf.int32, [None])
		self.action = tf.placeholder(tf.int32, [None])
		self.reward_to_go = tf.placeholder(tf.float32, [None])

		state_embedding = tf.concat([tf.nn.embedding_lookup(self.embedding, self.state),
		                             tf.nn.embedding_lookup(self.embedding, self.target)], axis=1)
		hidden = self.value_policy(state_embedding)
		value = self.value(hidden)
		with tf.variable_scope('new'):
			policy = self.policy(hidden)
		with tf.variable_scope('old'):
			policy_old = self.policy(hidden)
		assign_ops = []
		for new, old in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='new'),
		                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='old')):
			assign_ops.append(tf.assign(old, new))
		self.assign_ops = tf.group(*assign_ops)

		# use scaled std of embedding vectors as policy std
		sigma = tf.Variable(self.environment.sigma / 2.0, trainable=False, dtype=tf.float32)
		self.build_train(tf.nn.embedding_lookup(self.embedding, self.action), self.reward_to_go, value, policy, policy_old, sigma)
		self.decision = self.build_plan(policy, sigma)

	def build_train(self, action, reward_to_go, value, policy_mean, policy_mean_old, sigma):
		advantage = reward_to_go - tf.stop_gradient(value)
		# Gaussian policy with identity matrix as covariance mastrix
		ratio = tf.exp(0.5 * tf.reduce_sum(tf.square((action - policy_mean_old) * sigma), axis=-1) -
		               0.5 * tf.reduce_sum(tf.square((action - policy_mean) * sigma), axis=-1))
		surr_loss = tf.minimum(ratio * advantage, tf.clip_by_value(ratio, 1.0 - self.params.clip_epsilon, 1.0 + self.params.clip_epsilon) * advantage)
		surr_loss = -tf.reduce_mean(surr_loss, axis=-1)
		v_loss = tf.reduce_mean(tf.squared_difference(reward_to_go, value), axis=-1)

		optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
		self.step = optimizer.minimize(surr_loss + self.params.c_value * v_loss)

	def build_plan(self, policy_mean, sigma):
		policy = tf.distributions.Normal(policy_mean, sigma)
		action_embed = policy.sample()
		l2_diff = tf.squared_difference(tf.expand_dims(action_embed, axis=1),
		                                tf.nn.embedding_lookup(self.embedding, self.neighbors))
		decision = tf.argmin(tf.reduce_sum(l2_diff, axis=-1), axis=-1)
		del l2_diff
		gc.collect()
		return decision

	def value_policy(self, state):
		hidden = state
		for i, dim in enumerate(self.params.hidden_dim):
			hidden = fully_connected(hidden, dim, 'policy_value_' + str(i))
		return hidden

	def value(self, hidden):
		return fully_connected(hidden, 1, 'value_o', activation='linear')

	def policy(self, hidden):
		return fully_connected(hidden, self.params.embed_dim, 'policy_o', activation='linear')

	# the number of trajectories sampled is equal to batch size
	def collect_trajectory(self, sess):
		feed_state = self.environment.initial_state()
		states, actions = [], []
		for i in range(self.params.trajectory_length):
			states.append(deepcopy(feed_state))
			feed_neighor = self.environment.get_neighbors(feed_state[:, 0])
			# action contains indices of actual node IDs
			action_indices = sess.run(self.decision,
			                          feed_dict={self.state: feed_state[:, 0], self.target: feed_state[:, 1], self.neighbors: feed_neighor})
			action = feed_neighor[np.array(range(self.params.batch_size)), action_indices]
			actions.append(action)
			feed_state[:, 0] = action
			del feed_neighor
			gc.collect()
		states = np.transpose(np.array(states), axes=(1, 0, 2)).tolist()
		actions = np.transpose(np.array(actions)).tolist()
		return self.environment.reward_multiprocessing(states, actions)

	def train(self, sess):
		sess.run(tf.global_variables_initializer())
		for _ in tqdm(range(self.params.epoch), ncols=100):
			states, actions, rewards = self.collect_trajectory(sess)
			indices = range(self.params.trajectory_length * self.params.batch_size)
			shuffle(indices)
			batch_size = self.params.trajectory_length * self.params.batch_size / self.params.step
			for _ in tqdm(range(self.params.outer_step), ncols=100):
				for i in range(self.params.step):
					batch_indices = indices[i * batch_size : (i + 1) * batch_size]
					sess.run(self.step, feed_dict={self.state: states[batch_indices][:, 0], self.target: states[batch_indices][:, 1],
					                               self.action: actions[batch_indices], self.reward_to_go: rewards[batch_indices]})
			sess.run(self.assign_ops)
