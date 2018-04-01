from config import *
from Environment import *
from AutoPath import *

if __name__ == '__main__':
	environment = Environment(args)
	tf.reset_default_graph()
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
	with tf.device('/gpu:' + str(args.device_id)):
		agent = AutoPath(environment.params, environment)
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			agent.train(sess)
			print('Node type accuracy: %f' % agent.accuracy(sess))
			recommendation = agent.plan(sess)
			precision, recall = utils.precision_recall(recommendation, environment.test_data, args.top_k)
			print('Precision: %f, Recall %f' % (precision, recall))
