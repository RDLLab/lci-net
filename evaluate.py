from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import scipy.stats

from network_policy import NetworkPolicy

# fully observable
from fully_observable.agents.vin_baseline_agent import VINBaselineAgent
from fully_observable.agents.vin_lcinet_agent import VINLCINetAgent

from multi_iterator import MultiIterator

from arguments import parse_args
from env_parser import EnvParser


def print_results(results):
    results = np.expand_dims(results, axis=1)
    results = np.concatenate(results, axis=0)
    succ = results[:, 0]
    traj_len = results[succ > 0, 1]
    # collided = results[succ > 0, 2]
    collided = results[:, 2]
    reward = results[:, 3]
#    print ("Success rate: %.3f  Trajectory length: %.1f  Collision rate: %.3f"%(
#        np.mean(succ), np.mean(traj_len), np.mean(collided)))

    print("Success rate: %.3f   Trajectory length: %.1f +- %.3f   Collision rate: %.3f +- %0.3f" % (
        np.mean(succ), np.mean(traj_len), mean_confidence_interval(traj_len), np.mean(collided), mean_confidence_interval(collided)))


def mean_confidence_interval(data, confidence=0.95):
    # alpha = (1 - confidence) * 100
    # lower_p = alpha / 2.0
    # lower = max(0.0, np.percentile(data, lower_p))
    # upper_p = 100 - lower_p
    # upper = np.percentile(data, upper_p)
    # h = max(lower, upper)   # conservative approximation

    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)

    return h


def main(arglist):
    # get params from args and env file header
    params = parse_args(arglist)
    EnvParser.parse_input_header(params.path, params)
    print(params)

    # set K value if none given
    if params.K < 0:
        params.K = 3 * max(params.s_dims)

    # record = True
    record = False

    # here, load model path is root directory for all models
    modelfile_base = params.loadmodel[0] + "/baseline/final.chk"
    modelfile_transnet = params.loadmodel[0] + "/transnet/final.chk"
    modelfile_a2 = params.loadmodel[0] + "/transnet_a2_c8_050/final.chk"
    modelfile_a3 = params.loadmodel[0] + "/transnet_a3_c8_050/final.chk"
    modelfile_x = params.loadmodel[0] + "/transnet_x/final.chk"

    # fully observable
    modelfile_vin_base = params.loadmodel[0] + "/fo_baseline/final.chk"
    modelfile_vin_lcinet = params.loadmodel[0] + "/fo_lcinet/final.chk"

    #model_files = [modelfile_base, modelfile_transnet, modelfile_a2, modelfile_a3]
    #model_files = [modelfile_base, modelfile_transnet, modelfile_x]
    #model_files = [modelfile_base, modelfile_x]
    model_files = [modelfile_vin_base, modelfile_vin_lcinet]

    # built model into the default graph
    with tf.Graph().as_default():
        # Create a TF session
        #os.environ["CUDA_VISIBLE_DEVICES"] = ""  # use CPU
        sess = tf.Session(config=tf.ConfigProto())

        # build networks for evaluation
        #types = [QMDPNetBaseline, TransNet, TransNetA2, TransNetA3]
        #types = [QMDPNetBaseline, TransNet, TransNetX]
        #types = [QMDPNetBaseline, TransNetX]
        types = [VINBaselineAgent, VINLCINetAgent]
        policies = []

        for i in range(len(types)):
            scope_name = "n" + str(i)
            with tf.variable_scope(scope_name):
                # create network
                network = types[i](params, batch_size=1, step_size=1)
                network.build_inference()

                # map variable to this scope
                var_dict = {}
                for v in tf.trainable_variables(scope_name):
                    name = v.name.split('/', 1)[1]
                    var_dict[name.split(':')[0]] = v

                # load model variables
                saver = tf.train.Saver(var_list=var_dict)
                saver.restore(sess, model_files[i])

                # wrap network as policy
                policy = NetworkPolicy(network, sess)
                policies.append(policy)

        print("Evaluating %d environments, %d trajectories per environment, repeating simulation %d time(s)" %
              (params.eval_envs, params.eval_trajs, params.eval_repeats))
        eval_it = MultiIterator(params, params.env_type, policies, record)

        # run evaluation trials
        results = [[] for t in range(len(types) + 1)]
        for eval_i in range(params.eval_envs * params.eval_trajs * params.eval_repeats):
            res = eval_it.next()
            for t in range(len(types) + 1):
                results[t].append(res[t])

        if params.include_expert:
            print("Expert")
            print_results(results[0])
        for t in range(len(types)):
            print(str(types[t]).split('.')[-1].split('\'')[0])     # use simple class name
            print_results(results[t+1])


if __name__ == '__main__':
    main(sys.argv[1:])  # skip filename
