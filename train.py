"""
Script to train networks and evaluate the learned policy
"""

import os, sys
import numpy as np
import pkg_resources
try:
    tf_version = pkg_resources.get_distribution("tensorflow-gpu").version
except pkg_resources.DistributionNotFound:
    tf_version = pkg_resources.get_distribution("tensorflow").version
if tf_version[0] == '2':
    import tensorflow.compat.v1 as tf  # Use TF 1.X compatibility mode
    tf.disable_v2_behavior()
else:
    import tensorflow as tf
import scipy.stats

from datetime import datetime

from network_policy import NetworkPolicy
# fully observable
from fully_observable.agents.vin_baseline_agent import VINBaselineAgent
from fully_observable.agents.vin_lcinet_agent import VINLCINetAgent
# partially observable
from partially_obserable.agents.qn_baseline_agent import QNBaselineAgent
from partially_obserable.agents.qn_lcinet_agent import QNLCINetAgent

from arguments import parse_args
from eval_iterator import EvalIterator
from env_parser import EnvParser


def run_training(params, Network):
    """
    Train network via imitation learning.
    """

    # build data iterators
    parser = EnvParser(params.path, params)
    train_it = parser.get_training_iterator()
    valid_it = parser.get_validation_iterator()

    # built model into the default graph
    with tf.Graph().as_default():
        # build network for training
        network = Network(params, batch_size=params.batch_size, step_size=params.step_size)
        network.build_inference()  # build graph for inference including loss
        network.build_train(initial_lr=params.learning_rate)  # build training ops

        # Create a saver for writing training checkpoints.
        saver = tf.compat.v1.train.Saver(var_list=tf.trainable_variables(), max_to_keep=500)  # if max_to_keep=0 will output useless log info

        # Get initialize Op
        init = tf.global_variables_initializer()

        # Create a TF session
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Run the Op to initialize variables
        sess.run(init)

        # load previously saved model
        if params.loadmodel:
            print("Loading from "+params.loadmodel[0])
            loader = tf.train.Saver(var_list=tf.trainable_variables())
            loader.restore(sess, params.loadmodel[0])

        summary_writer = tf.compat.v1.summary.FileWriter(params.logpath, sess.graph)
        summary_writer.flush()

    epoch = -1
    best_epoch = 0
    no_improvement_epochs = 0
    patience = params.patience_first  # initial patience
    decay_step = 0
    valid_losses = []

    if params.print_timing:
        print(datetime.now())

    for epoch in range(params.epochs):
        train_it.reset()
        training_loss = 0.0
        step_t = 0
        # for step in range(train_feed.steps_in_epoch):
        while not train_it.is_finished():
            data = train_it.next()

            feed_dict = {network.placeholders[i]: data[i] for i in range(len(network.placeholders))}

            _, loss, _ = sess.run([network.train_op, network.loss, network.update_belief_op],
                                  feed_dict=feed_dict)
            training_loss += loss

            step_t += 1
            if params.verbose_training and step_t % 100 == 0:
                print("Step " + str(step_t))

        # save belief and restore it after validation
        belief = sess.run([network.belief])[0]

        # accumulate loss over the entire validation set
        valid_it.reset()
        valid_loss = 0.0
        step_v = 0
        # for step in range(valid_feed.steps_in_epoch):  # params.validbatchsize
        while not valid_it.is_finished():
            data = valid_it.next()

            # assert step > 0 or np.isclose(data[3], 1.0).all()
            feed_dict = {network.placeholders[i]: data[i] for i in range(len(network.placeholders))}
            loss, _ = sess.run([network.loss, network.update_belief_op], feed_dict=feed_dict)
            valid_loss += loss

            step_v += 1
            if params.verbose_training and step_v % 100 == 0:
                print("Step " + str(step_v))

        tf.assign(network.belief, belief)

        training_loss /= step_t
        valid_loss /= step_v

        # print status
        lr = sess.run([network.learning_rate])[0]
        print('Epoch %d, lr=%f, training loss=%.3f, valid loss=%.3f' % (epoch, lr, training_loss, valid_loss))
        if params.print_timing:
            print(datetime.now())

        valid_losses.append(valid_loss)
        best_epoch = np.array(valid_losses).argmin()

        # save a checkpoint if needed
        if best_epoch == epoch or epoch == 0:
            best_model = saver.save(sess, os.path.join(params.logpath, 'model.chk'), global_step=epoch)
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        # check for early stopping
        if no_improvement_epochs > patience:
            # finish training if learning rate decay steps reached
            if decay_step >= params.decaystep:
                break
            decay_step += 1
            no_improvement_epochs = 0

            # restore best model found so far
            saver.restore(sess, best_model)

            # decay learning rate
            sess.run(tf.assign(network.decay_step, decay_step))
            learning_rate = network.learning_rate.eval(session=sess)
            print("Decay step %d, lr = %f" % (decay_step, learning_rate))

            # use smaller patience for future iterations
            patience = params.patience_rest

    # Training done
    epoch += 1
    print("Training loop over after %d epochs" % epoch)

    # restore best model
    if best_epoch != epoch:
        print("Restoring %s from epoch %d" % (str(best_model), best_epoch))
        saver.restore(sess, best_model)

    # save best model
    checkpoint_file = os.path.join(params.logpath, 'final.chk')
    saver.save(sess, checkpoint_file)

    return checkpoint_file


def run_eval(params, modelfile, Network):
    # built model into the default graph
    with tf.Graph().as_default():
        # build network for evaluation
        network = Network(params, batch_size=1, step_size=1)
        network.build_inference()

        # Create a saver for loading checkpoint
        saver = tf.train.Saver(var_list=tf.trainable_variables())

        # Create a TF session
        os.environ["CUDA_VISIBLE_DEVICES"] = "" # use CPU
        sess = tf.Session(config=tf.ConfigProto())

        # load model from file
        saver.restore(sess, modelfile)

        # policy
        policy = NetworkPolicy(network, sess)

    print("Evaluating %d environments, %d trajectories per environment, repeating simulation %d time(s)" %
          (params.eval_envs, params.eval_trajs, params.eval_repeats))
    eval_it = EvalIterator(params, params.env_type, policy)
    expert_res = []
    network_res = []
    for eval_i in range(params.eval_envs * params.eval_trajs * params.eval_repeats):
        res = eval_it.next()
        expert_res.append(res[:1][0])
        network_res.append(res[1:][0])
    if params.include_expert:
        print("Expert")
        print_results(expert_res)
    print("Network type " + params.network_type)
    print_results(network_res)


def print_results(results):
    results = np.expand_dims(results, axis=1)
    results = np.concatenate(results, axis=0)
    succ = results[:, 0]
    traj_len = results[succ > 0, 1]
    collided = results[:, 2]

    print("Success rate: %.3f   Trajectory length: %.1f +- %.3f   Collision rate: %.3f +- %0.3f" % (
        np.mean(succ), np.mean(traj_len), mean_confidence_interval(traj_len), np.mean(collided), mean_confidence_interval(collided)))


def mean_confidence_interval(data, confidence=0.95):
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

    # set network type
    network_dict = {
        'vin_baseline': VINBaselineAgent,
        'vin_lcinet': VINLCINetAgent,
        'qmdpnet_baseline': QNBaselineAgent,
        'qmdpnet_lcinet': QNLCINetAgent
    }
    Network = network_dict[params.network_type]

    if params.epochs == 0:
        assert len(params.loadmodel) == 1
        modelfile = params.loadmodel[0]
    else:
        modelfile = run_training(params, Network)

    run_eval(params, modelfile, Network)


if __name__ == '__main__':
    main(sys.argv[1:])  # skip filename
