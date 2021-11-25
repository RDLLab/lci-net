import argparse
from utils.dotdict import dotdict


def parse_args(arglist):

    parser = argparse.ArgumentParser(description='Run training on gridworld')

    parser.add_argument('path',
                        help='Path to data folder containing train and test subfolders')
    parser.add_argument('--logpath', default='./log/',
                        help='Path to save log and trained model')

    parser.add_argument('--loadmodel', nargs='*',
                        help='Load model weights from checkpoint')

    parser.add_argument('--eval_envs', type=int,
                        default=20,
                        help='Number of environments to evaluate the learned policy on')
    parser.add_argument('--eval_trajs', type=int,
                        default=5,
                        help='Number of trajectories per environment to evaluate the learned policy on')
    parser.add_argument('--eval_repeats', type=int,
                        default=1,
                        help='Repeat simulating policy for a given number of times. Use 5 for stochastic domains')

    parser.add_argument('--batch_size', type=int, default=100,
                        help='Size of minibatches for training')
    parser.add_argument('--training_envs', type=float, default=0.9,
                        help='Proportion of training data used for training. Remainder will be used for validation')
    parser.add_argument('--step_size', type=int, default=4,
                        help='Number of maximum steps for backpropagation through time')
    parser.add_argument('--lim_traj_len', type=int, default=100,
                        help='Clip trajectories to a maximum length')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--patience_first', type=int,
                        default=30,
                        help='Start decaying learning rate if no improvement for a given number of steps')
    parser.add_argument('--patience_rest', type=int,
                        default=5,
                        help='Patience after decay started')
    parser.add_argument('--decaystep', type=int,
                        default=15,
                        help='Total number of learning rate decay steps')
    parser.add_argument('--epochs', type=int,
                        default=1000,
                        help='Maximum number of epochs')

    parser.add_argument('--cache', nargs='*',
                        default=['steps', 'envs', 'bs'],
                        help='Cache nodes from pytable dataset. Default: steps, envs, bs')

    parser.add_argument('-K', '--K', type=int,
                        default=-1,
                        help='Number of iterations of value iteration in QMDPNet/TransNet. Compute from grid size if negative.')

    parser.add_argument('--network_type', default='base',
                        help='Type of architecture to be used for the agent.')

    parser.add_argument('--include_expert', action='store_true',
                        help='Include expert in evaluation comparison.')

    parser.add_argument('--print_timing', action='store_true',
                        help='Print time after each epoch.')

    parser.add_argument('--verbose_training', action='store_true',
                        help='Show intermediate step count during training')

    parser.add_argument('--num_classes', type=int,
                        default=16,
                        help='Force number of classes (transnet-aX only)')

    parser.add_argument('--class_weight', type=float,
                        default=0.5,
                        help='Weight for class term in loss function (transnet-a2/a3 only')

    parser.add_argument('--obs_range', type=int,
                        default=3,
                        help='Width of kernel in first convolution layer of f_Z (i.e. the range within which theta can influence f_Z).')

    parser.add_argument('--t_range', type=int,
                        default=3,
                        help='Width of kernel in first convolution layer of f_T (i.e. the range within which theta can influence f_T).')

    parser.add_argument('--use_simple_dirs', action='store_true',
                        help='Constrain transition model to simple directions only')

    parser.add_argument('--use_simple_features', action='store_true',
                        help='Constrain convolution window to simple directions only')

    args = parser.parse_args(args=arglist)

    # load domain parameters
    params = dotdict(vars(args))

    return params
