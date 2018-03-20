from config import *
from Environment import *
from PPO import *

if __name__ == '__main__':
	environment = Environment(args)
	tf.reset_default_graph()
	agent = PPO(environment.params, environment)
	with tf.Session() as sess:
		agent.train(sess)
