from config import *
from Environment import *
from PPO import *

if __name__ == '__main__':
	environment = Environment(args)
	tf.reset_default_graph()
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
	with tf.device('/gpu:' + str(args.device_id)):
		agent = PPO(environment.params, environment)
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			agent.train(sess)
			environment.dump_embed(sess.run(agent.embedding))
