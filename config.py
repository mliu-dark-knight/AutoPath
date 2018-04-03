import argparse
import os


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device_id', type=int, default=0, help=None)
	parser.add_argument('--max_neighbor', type=int, default=20, help=None)
	parser.add_argument('--num_node', type=int, default=-1, help='Total number of nodes')
	parser.add_argument('--num_attribute', type=int, default=-1, help='Total number of attributes')
	parser.add_argument('--num_type', type=int, default=-1, help='Total number of node types')
	parser.add_argument('--embed_dim', type=int, default=16, help=None)
	parser.add_argument('--hidden_dim', type=list, default=[32], help=None)
	parser.add_argument('--learning_rate', type=float, default=1e-3, help=None)
	parser.add_argument('--keep_prob', type=float, default=1.0, help='Used for dropout')
	parser.add_argument('--clip_epsilon', type=float, default=1e-1, help=None)
	parser.add_argument('--c_value', type=float, default=1.0, help='Coefficient for value function loss')
	parser.add_argument('--batch_size', type=int, default=10, help='Number of trajectories sampled')
	parser.add_argument('--trajectory_length', type=int, default=5, help=None)
	parser.add_argument('--epoch', type=int, default=2, help=None)
	parser.add_argument('--classification_step', type=int, default=1, help='Number of rounds of mini batch SGD per epoch for classification')
	parser.add_argument('--PPO_step', type=int, default=2, help='Number of rounds of mini batch SGD per epoch for PPO')
	parser.add_argument('--step', type=int, default=2, help=None)
	parser.add_argument('--top_k', type=int, default=5, help='Top k for precision and recall')
	return parser.parse_args()


def init_dir(args):
	args.data_dir = os.getcwd() + '/data/'
	args.node_file = args.data_dir + '/node.txt'
	args.link_file = args.data_dir + '/link.txt'
	args.train_files = [args.data_dir + '/train_1.txt', args.data_dir + '/train_2.txt']
	args.test_file = args.data_dir + '/test.txt'

args = parse_args()
init_dir(args)
