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
import numpy as np


class NetworkPolicy():
    """
    POMDP Policy wrapper for network. Implements two functions: reset and eval.
    """
    def __init__(self, network, sess):
        self.network = network
        self.sess = sess

        self.belief_img = None
        self.env_img = None
        self.goal_img = None

        assert self.network.batch_size == 1 and self.network.step_size == 1

    def reset(self, env_img, goal_img, belief_img):
        #TODO
        """

        :param env_img:
        :param goal_img:
        :param belief_img:
        :return:
        """
        s_dims = self.network.params.s_dims

        self.belief_img = belief_img.reshape([1] + s_dims)
        self.env_img = env_img.reshape([1] + s_dims)
        self.goal_img = goal_img.reshape([1] + s_dims)

        self.sess.run(tf.assign(self.network.belief, self.belief_img))
        #
        # feed_dict = {tf.get_default_graph().get_tensor_by_name('In_map:0'): env_img,
        #              tf.get_default_graph().get_tensor_by_name('In_goal:0'): goal_img}
        # self.context_value = self.sess.run(self.network.context_variables, feed_dict=feed_dict)

    def eval(self, last_act, last_obs):
        #TODO
        """

        :param last_act:
        :param last_obs:
        :return:
        """
        isstart = np.array([0])
        last_act = np.reshape(last_act, [1, 1])
        last_obs = np.reshape(last_obs, [1, 1, self.network.params.obs_len])

        # input data. do not need weight and label for prediction
        data = [self.env_img, self.goal_img, self.belief_img, isstart, last_act, last_obs]
        feed_dict = {self.network.placeholders[i]: data[i] for i in range(len(self.network.placeholders)-2)}

        # evaluate QMDPNet
        logits, _ = self.sess.run([self.network.logits, self.network.update_belief_op], feed_dict=feed_dict)
        act = np.argmax(logits.flatten())

        return act