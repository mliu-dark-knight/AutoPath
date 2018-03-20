import argparse
import os


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--debug', type=bool, default=False, help=None)
	parser.add_argument('--num_node', type=int, default=-1, help='Total number of nodes')
	parser.add_argument('--node_type', type=list, default=['user', 'business', 'star', 'category', 'city'], help=None)
	parser.add_argument('--embed_dim', type=int, default=64, help=None)
	parser.add_argument('--hidden_dim', type=list, default=[256], help=None)
	parser.add_argument('--learning_rate', type=float, default=1e-3, help=None)
	parser.add_argument('--clip_epsilon', type=float, default=1e-1, help=None)
	parser.add_argument('--c_value', type=float, default=1.0, help='Coefficient for value function loss')
	parser.add_argument('--batch_size', type=int, default=4, help='Number of trajectories sampled')
	parser.add_argument('--trajectory_length', type=int, default=2, help=None)
	parser.add_argument('--epoch', type=int, default=4, help=None)
	parser.add_argument('--outer_step', type=int, default=4, help='Number of rounds of mini batch SGD per epoch')
	parser.add_argument('--step', type=int, default=2, help=None)
	parser.add_argument('--num_process', type=int, default=4, help='Number of subprocesses')
	return parser.parse_args()


def init_dir(args):
	args.data_dir = os.getcwd() + '/data/'
	args.pair_file = args.data_dir + '/link.txt'

args = parse_args()
init_dir(args)