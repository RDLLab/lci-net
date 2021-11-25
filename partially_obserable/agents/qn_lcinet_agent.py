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

from conv_nd import conv_layers
from linear import fc_layers


class QNLCINetAgent:
    """
    Class implementing agent based on LCI-Net embedded in QMDP-net architecture.
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

        self.params.batch_size = batch_size

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
        isstart = tf.to_float(isstart)
        isstart = tf.reshape(isstart, [self.batch_size] + [1]*(b0.get_shape().ndims-1))
        act_in = tf.to_int32(act_in)
        obs_in = tf.to_float(obs_in)
        act_label = tf.to_int32(act_label)

        outputs = []

        # pre-compute context, fixed through time
        with tf.variable_scope("planner"):
            Q = PlannerNet.VI(map, goal, self.params)
        with tf.variable_scope("filter"):
            Z = FilterNet.f_Z(map, self.params)

        self.context_tensors = [Q, Z]

        # create variable for hidden belief (equivalent to the hidden state of an RNN)
        self.belief = tf.Variable(np.zeros(b0.get_shape().as_list(), 'f'), trainable=False, name="hidden_belief")

        # figure out current b. b = b0 if isstart else blast
        b = (b0 * isstart) + (self.belief * (1-isstart))

        for step in range(self.step_size):
            # filter
            with tf.variable_scope("filter") as step_scope:
                if step >= 1:
                    step_scope.reuse_variables()
                b = FilterNet.beliefupdate(map, Z, b, act_in[step], obs_in[step], self.params)

            # planner
            with tf.variable_scope("planner") as step_scope:
                if step >= 1:
                    step_scope.reuse_variables()
                action_pred = PlannerNet.policy(Q, b, self.params)
                outputs.append(action_pred)

        # create op that updates the belief
        self.update_belief_op = self.belief.assign(b)

        # compute loss (cross-entropy)
        logits = tf.stack(values=outputs, axis=0)  # shape is [step_size, batch_size, num_action]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=act_label)

        # weight loss. weights are 0.0 for steps after the end of a trajectory, otherwise 1.0
        loss = loss * weight
        loss = tf.reduce_mean(loss, axis=[0, 1], name='xentropy')

        self.logits = logits
        self.loss = loss

    def build_train(self, initial_lr):
        """
        Creates decay step, exponential decay learning rate, training op
        :param initial_lr: initial learning rate
        :return: None
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
    def f_R(map, goal, num_action, s_dim):
        theta = tf.stack([map, goal], axis=(s_dim + 1))
        R = conv_layers(theta, np.array([[3, 150, 'relu'], [1, num_action, 'lin']]), "R_conv")
        return R

    @staticmethod
    def VI(map, goal, params):
        """
        Builds neural network implementing VI. this is the first part of planner module. Fixed through time.
        inputs: map (batch x N x N ...) and goal(batch x N x N ...)
        returns: Q_K
        """
        s_dim = len(params.s_dims)

        # build reward model R
        R = PlannerNet.f_R(map, goal, params.num_action, s_dim)

        # get transition model T. It represents the transition model in the filter, but the weights are not shared.
        features = build_shifts([], [-(params.t_range // 2) for _ in range(s_dim)],
                                [(params.t_range // 2) + 1 for _ in range(s_dim)],
                                params.use_simple_features)
        shifts = build_shifts([], [-(params.k_dims[i] // 2) for i in range(s_dim)],
                              [(params.k_dims[i]) // 2 + 1 for i in range(s_dim)],
                              params.use_simple_dirs)
        T = FilterNet.f_T(params, map, features, shifts)

        # initialize value image
        V = tf.zeros(map.get_shape().as_list() + [1])
        Q = None

        # repeat value iteration K times
        for i in range(params.K):
            # apply transition and sum
            V_prime = tf.stack([apply_shift(V, f, params) for f in shifts], axis=-1)
            V_prime = tf.reshape(V_prime, [-1] + params.s_dims + [1, len(shifts)])
            Q = tf.reduce_sum(V_prime * T, axis=-1)
            Q = Q + R
            V = tf.reduce_max(Q, axis=[s_dim + 1], keepdims=True)

        return Q

    @staticmethod
    def f_pi(q, num_action):
        action_pred = fc_layers(q, np.array([[num_action, 'lin']]), names="pi_fc")
        return action_pred

    @staticmethod
    def policy(Q, b, params):
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


class FilterNet():
    @staticmethod
    def f_Z(map, params):
        """
        This implements f_Z, outputs an observation model (Z). Fixed through time.
        inputs: map (batch x N x N ... array)
        returns: Z
        """
        # CNN: theta -> Z
        map = tf.expand_dims(map, -1)
        Z = conv_layers(map, np.array([[params.obs_range, 150, 'lin'], [1, params.num_obs, 'sig']]), "Z_conv")

        # normalize over observations
        Z_sum = tf.reduce_sum(Z, [len(params.s_dims) + 1], keepdims=True)
        Z = tf.math.divide(Z, Z_sum + 1e-8)  # add a small number to avoid division by zero

        return Z

    @staticmethod
    def f_A(action, num_action):
        # identity function
        w_A = tf.one_hot(action, num_action)
        return w_A

    @staticmethod
    def f_O(local_obs, params):
        w_O = fc_layers(local_obs, np.array([[params.num_obs, 'tanh'], [params.num_obs, 'smax']]), names="O_fc")
        return w_O

    @staticmethod
    def f_T(params, map, features, shifts):
        """
        Transition model (T), which gives the probability of transition into each relative position for each cell in
        the map.

        :param params:
        :param map:
        :param features:
        :param shifts:
        :return: T
        """
        s_dim = len(params.s_dims)
        hidden_size = len(shifts)   # fix hidden size to be equal to number of shifts (for backwards compatibility)

        map_prime = tf.stack([apply_shift(map, f, params) for f in features], axis=-1)

        # weights and biases for 1st transition layer
        init_w = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32)
        w_t1 = tf.get_variable("w_T1", shape=[len(features), hidden_size], dtype=tf.float32,
                               initializer=init_w)
        init_b = tf.constant(0., shape=[hidden_size], dtype=tf.float32)
        b_t1 = tf.get_variable("b_T1", dtype=tf.float32, initializer=init_b)

        # fully connected cluster to produce intermediate layer
        map_inter = tf.stack(
            [tf.reduce_sum(map_prime * tf.reshape(w_t1[:, i], [1 for _ in range(s_dim + 1)] + [-1]), axis=-1)
             for i in range(hidden_size)], axis=-1)
        map_inter += tf.reshape(b_t1, [1 for _ in range(s_dim + 1)] + [-1])
        map_inter = tf.nn.relu(map_inter)

        # weights and biases for 2nd transition layer
        w_t2 = tf.get_variable("w_T2", shape=[hidden_size, len(shifts) * params.num_action],
                               dtype=tf.float32, initializer=init_w)
        init_b2 = tf.constant(0., shape=[len(shifts) * params.num_action], dtype=tf.float32)
        b_t2 = tf.get_variable("b_T2", dtype=tf.float32, initializer=init_b2)

        # fully connected cluster to produce transition weights
        T = tf.stack(
            [tf.reduce_sum(map_inter * tf.reshape(w_t2[:, i], [1 for _ in range(s_dim + 1)] + [-1]), axis=-1)
             for i in range(len(shifts) * params.num_action)], axis=-1)
        T += tf.reshape(b_t2, [1 for _ in range(s_dim + 1)] + [-1])
        T = tf.reshape(T, [params.batch_size] + params.s_dims + [params.num_action, len(shifts)])
        T = tf.nn.softmax(T, axis=-1)

        return T

    @staticmethod
    def beliefupdate(map, Z, b, action, local_obs, params):
        """
        Belief update in the filter module with pre-computed Z.
        :param map:
        :param Z:
        :param b: belief (b_i), [batch, N, M, 1]
        :param action: action input (a_i)
        :param local_obs: observation input (o_i)
        :param params:
        :return: updated belief b_(i+1)
        """
        s_dim = len(params.s_dims)

        # step 1: update belief with transition
        # get transition model (T)
        features = build_shifts([], [-(params.t_range // 2) for _ in range(s_dim)],
                                [(params.t_range // 2) + 1 for _ in range(s_dim)],
                                params.use_simple_features)
        shifts = build_shifts([], [-(params.k_dims[i] // 2) for i in range(s_dim)],
                              [(params.k_dims[i]) // 2 + 1 for i in range(s_dim)],
                              params.use_simple_dirs)
        T = FilterNet.f_T(params, map, features, shifts)

        # apply convolution which corresponds to the transition function in an MDP (f_T)
        b = tf.expand_dims(b, -1)
        b_prime = tf.stack([apply_shift(b, f, params)
                            for f in shifts], axis=-1)
        b_prime = tf.reshape(b_prime, [-1] + params.s_dims + [1, len(shifts)])
        b_prime = tf.reduce_sum(b_prime * T, axis=-1)

        # index into the appropriate channel of b_prime
        w_A = FilterNet.f_A(action, params.num_action)
        for _ in range(s_dim):
            w_A = tf.expand_dims(w_A, 1)
        b_prime_a = tf.reduce_sum(tf.multiply(b_prime, w_A), [s_dim + 1], keepdims=False) # soft indexing

        # step 2: update belief with observation
        # get observation probabilities for the observation input by soft indexing
        w_O = FilterNet.f_O(local_obs, params)
        for _ in range(s_dim):
            w_O = tf.expand_dims(w_O, 1)
        Z_o = tf.reduce_sum(tf.multiply(Z, w_O), [s_dim + 1], keepdims=False)   # soft indexing

        b_next = tf.multiply(b_prime_a, Z_o)

        # step 3: normalize over the state space
        # add small number to avoid division by zero
        b_next = tf.math.divide(b_next, tf.reduce_sum(b_next, [i for i in range(1, s_dim + 1)], keepdims=True) + 1e-8)

        return b_next


def build_shifts(features, lower, upper, use_simple_dirs):
    """
    Helper method for building a list of shift directions. If use_simple_dirs is False, covers every combination of
    relative position changes between lower and upper in each dimension. Otherwise, covers orthogonal combinations only.
    """
    assert len(lower) == len(upper)
    if len(lower) == 0:
        # base case
        return features
    # recursive case
    current_lower = lower[0]
    new_lower = lower[1:]
    current_upper = upper[0]
    new_upper = upper[1:]

    if len(features) == 0:
        # no existing features
        features_new = []
        for i in range(current_lower, current_upper):
            features_new.append([i])
    else:
        # add to existing features
        if use_simple_dirs:
            # apply simple shifts only
            features_new = []
            for f in features:
                f_is_empty = True
                for x in f:
                    if x != 0:
                        f_is_empty = False

                if f_is_empty:
                    for i in range(current_lower, current_upper):
                        features_new.append(f + [i])
                else:
                    features_new.append(f + [0])
        else:
            # apply all possible shifts
            features_new = []
            for f in features:
                for i in range(current_lower, current_upper):
                    features_new.append(f + [i])
    return build_shifts(features_new, new_lower, new_upper, use_simple_dirs)


def apply_shift(map, shift, params):
    """
    Helper method for applying a shift (1 change in position for each dimension) to a given tensor. Number of dimensions
    in map must match length of shift.
    """
    rolled_m = map
    for dim in range(len(params.s_dims)):
        if abs(shift[dim]) > 0:
            s = -1 * shift[dim]
            a = dim + 1
            rolled_m = tf.roll(rolled_m, shift=s, axis=a)
    return rolled_m




