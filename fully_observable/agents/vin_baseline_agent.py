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

from conv_nd import conv_nd, conv_layers
from linear import fc_layers


class VINBaselineAgent:
    """
    Class implementing a QMDP-Net for the grid navigation domain
    """

    def __init__(self, params, batch_size=1, step_size=1):
        """
        :param params: dotdict describing the domain and network hyperparameters
        :param batch_size: minibatch size for training. Use batch_size=1 for evaluation
        :param step_size: limit the number of steps for backpropagation through time. Use step_size=1 for evaluation.
        """
        self.params = params
        self.batch_size = batch_size
        self.step_size = step_size

        self.placeholders = None
        self.context_tensors = None
        self.belief = None
        self.update_belief_op = None
        self.logits = None
        self.loss = None

        self.action_pred = None

        self.decay_step = None
        self.learning_rate = None
        self.train_op = None

        self.kp_out = None   # debug: for outputting learned planner T model
        self.kf_out = None  # debug: for outputting learned filter T model
        self.v_out = None  # debug: for outputting learned V
        self.r_out = None  # debug: for outputting learned R

    def build_placeholders(self):
        """
        Creates placeholders for all inputs in self.placeholders
        """
        s_dims = self.params.s_dims
        obs_len = self.params.obs_len
        step_size = self.step_size
        batch_size = self.batch_size

        placeholders = []
        placeholders.append(tf.placeholder(tf.uint8,
                                            shape=tuple([batch_size] + s_dims), name='In_map'))

        placeholders.append(tf.placeholder(tf.uint8,
                                            shape=tuple([batch_size] + s_dims), name='In_goal'))

        placeholders.append(tf.placeholder(tf.float32,
                                            shape=tuple([batch_size] + s_dims),
                                            name='In_b0'))

        placeholders.append(tf.placeholder(tf.uint8,
                                            shape=(batch_size,), name='In_isstart'))

        placeholders.append(tf.placeholder(tf.uint8,
                                            shape=(step_size, batch_size), name='In_actions'))

        placeholders.append(tf.placeholder(tf.uint8,
                                            shape=(step_size, batch_size, obs_len), name='In_local_obs'))

        placeholders.append(tf.placeholder(tf.float32,
                                            shape=(step_size, batch_size), name='In_weights'))

        placeholders.append(tf.placeholder(tf.uint8,
                                            shape=(step_size, batch_size), name='Label_a'))

        self.placeholders = placeholders

    def build_placeholders_rl(self):
        """
        Create placeholders for all inputs for reinforcement learning in self.placeholders
        """
        s_dims = self.params.s_dims
        obs_len = self.params.obs_len
        batch_size = self.batch_size

        self.map_ph = tf.placeholder(tf.uint8, shape=tuple([batch_size] + s_dims), name='In_map')
        self.goal_ph = tf.placeholder(tf.uint8, shape=tuple([batch_size] + s_dims), name='In_goal')
        self.b0_ph = tf.placeholder(tf.float32, shape=tuple([batch_size] + s_dims), name='In_b0')
        self.isstart_ph = tf.placeholder(tf.uint8, shape=(batch_size,), name='In_isstart')
        self.in_act_ph = tf.placeholder(tf.uint8, shape=(batch_size,), name='In_actions')
        self.in_obs_ph = tf.placeholder(tf.uint8, shape=(batch_size, obs_len), name='In_local_obs')
        self.per_a_ph = tf.placeholder(tf.uint8, shape=(batch_size,), name='Performed_action')
        self.reward_ph = tf.placeholder(tf.float32, shape=(batch_size,), name='Reward')
        self.b_tgt_ph = tf.placeholder(tf.float32, shape=tuple([batch_size] + s_dims), name='B_target')
        self.weight_ph = tf.placeholder(tf.uint8, shape=(batch_size,), name='Weight')

        self.placeholders = [self.map_ph, self.goal_ph, self.b0_ph, self.isstart_ph, self.in_act_ph,
                             self.in_obs_ph, self.weight_ph, self.per_a_ph, self.reward_ph, self.b_tgt_ph]

    def build_inference(self, reuse=False):
        """
        Creates placeholders, ops for inference and loss
        Unfolds filter and planner through time
        Also creates an op to update the belief. It should be always evaluated together with the loss.
        :param reuse: reuse variables if True
        :return: None
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        self.build_placeholders()

        map, goal, b0, isstart, act_in, obs_in, weight, act_label = self.placeholders

        # types conversions
        map = tf.to_float(map)
        goal = tf.to_float(goal)
        obs_in = tf.to_int32(obs_in)
        act_label = tf.to_int32(act_label)

        outputs = []

        # pre-compute context, fixed through time
        with tf.variable_scope("planner"):
            Q, self.v_out, self.r_out, self.kp_out = PlannerNet.VI(map, goal, self.params)

        self.context_tensors = [Q]

        # dummy variable required for compatibility with batch_train_v2
        self.belief = tf.Variable(np.zeros(b0.get_shape().as_list(), 'f'), trainable=False, name="hidden_belief")

        for step in range(self.step_size):
            # set b to be one hot of the current state (as revealed by obs in)
            b = tf.ones(shape=np.shape(map))
            for i in range(len(self.params.s_dims)):
                # get one hot index for this dimension
                idx = obs_in[step, :, i]    # dims = (b)
                idx_one_hot = tf.one_hot(idx, self.params.s_dims[i])    # dims = (b, s_dims[i])

                # stack to fit dimensions of state space
                for _ in range(0, i):
                    idx_one_hot = tf.expand_dims(idx_one_hot, axis=1)
                for _ in range(i+1, len(self.params.s_dims)):
                    idx_one_hot = tf.expand_dims(idx_one_hot, axis=-1)

                # multiply belief by this coordinate index
                b *= idx_one_hot

            # planner
            with tf.variable_scope("planner") as step_scope:
                if step >= 1:
                    step_scope.reuse_variables()
                action_pred = PlannerNet.policy(Q, b, self.params)
                outputs.append(action_pred)

        # create op that updates the belief (for fully observable, does nothing)
        self.update_belief_op = tf.no_op()

        # compute loss (cross-entropy)
        logits = tf.stack(values=outputs, axis=0)  # shape is [step_size, batch_size, num_action]

        # logits = tf.reshape(logits, [self.step_size*self.batch_size, self.params.num_action])
        # act_label = tf.reshape(act_label, [-1])
        # weight = tf.reshape(weight, [-1])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=act_label)

        # weight loss. weights are 0.0 for steps after the end of a trajectory, otherwise 1.0
        loss = loss * weight
        loss = tf.reduce_mean(loss, axis=[0, 1], name='xentropy')

        self.logits = logits
        self.loss = loss

    def build_train(self, initial_lr):
        """
        """

        # Decay learning rate by manually incrementing decay_step
        decay_step = tf.Variable(0.0, name='decay_step', trainable=False)
        learning_rate = tf.compat.v1.train.exponential_decay(
            initial_lr, decay_step, 1, 0.8, staircase=True, name="learning_rate")

        trainable_variables = tf.compat.v1.trainable_variables()

        optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate, decay=0.9)
        # clip gradients
        grads = tf.gradients(self.loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0, use_norm=tf.global_norm(grads))

        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        self.decay_step = decay_step
        self.learning_rate = learning_rate
        self.train_op = train_op


class PlannerNet:
    @staticmethod
    def f_R(map, goal, num_action, s_dim, legacy_mode):
        theta = tf.stack([map, goal], axis=(s_dim + 1))
        R = conv_layers(theta, np.array([[3, 150, 'relu'], [1, num_action, 'lin']]), "R_conv", legacy_mode)
        return R

    @staticmethod
    def f_T(params):
        # get transition kernel
        initializer = tf.truncated_normal_initializer(mean=1.0/9.0, stddev=1.0/90.0, dtype=tf.float32)
        kernel = tf.get_variable("w_T_conv", [np.prod(params.k_dims), params.num_action], initializer=initializer, dtype=tf.float32)

        # enforce proper probability distribution (i.e. values must sum to one) by softmax
        kernel = tf.nn.softmax(kernel, axis=0)
        kernel = tf.reshape(kernel, params.k_dims + [1, params.num_action], name="T_w")

        return kernel

    @staticmethod
    def VI(map, goal, params):
        """
        builds neural network implementing value iteration. this is the first part of planner module. Fixed through time.
        inputs: map (batch x N x N) and goal(batch)
        returns: Q_K, and optionally: R, list of Q_i
        """
        s_dim = len(params.s_dims)

        # build reward model R
        R = PlannerNet.f_R(map, goal, params.num_action, s_dim, params.legacy_mode)

        # get transition model Tprime. It represents the transition model in the filter, but the weights are not shared.
        kernel = PlannerNet.f_T(params)

        # initialize value image
        V = tf.zeros(map.get_shape().as_list() + [1])
        Q = None

        # repeat value iteration K times
        for i in range(params.K):
            # apply transition and sum
            Q = conv_nd(V, kernel, params.legacy_mode)
            Q = Q + R
            V = tf.reduce_max(Q, axis=[s_dim + 1], keepdims=True)

        return Q, V, R, kernel

    @staticmethod
    def f_pi(q, num_action):
        action_pred = fc_layers(q, np.array([[num_action, 'lin']]), names="pi_fc")
        return action_pred

    @staticmethod
    def policy(Q, b, params, reuse=False):
        """
        second part of planner module
        :param Q: input Q_K after value iteration
        :param b: belief at current step
        :param params: params
        :return: a_pred,  vector with num_action elements, each has the
        """
        s_dim = len(params.s_dims)
        # weight Q by the belief
        b_tiled = tf.tile(tf.expand_dims(b, s_dim + 1), [1 for _ in range(s_dim + 1)] + [params.num_action])
        q = tf.multiply(Q, b_tiled)
        # sum over states
        q = tf.reduce_sum(q, [i + 1 for i in range(s_dim)], keepdims=False)

        # low-level policy, f_pi
        action_pred = PlannerNet.f_pi(q, params.num_action)

        return action_pred



