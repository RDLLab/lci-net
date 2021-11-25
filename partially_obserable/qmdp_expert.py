import numpy as np


class QMDPExpert:
    """
    Class representing a QMDP-based action selection expert. V2 - N-dimensional compatibility

    This class is an expert.
    """

    GAMMA = 0.99
    NUM_ITERATIONS = 30

    def __init__(self, state_shape, action_shape, obs_shape, transition_model, reward_model, observation_model,
                 b_initial=None, get_true_state=None):
        """
        Perform value iteration on the underlying MDP and initialise belief state.
        """
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.obs_shape = obs_shape
        self.obs_size = np.size(obs_shape)
        self.num_obs = np.prod(obs_shape)
        self.num_iterations = max(state_shape) * (len(state_shape) + 1)
        # self.num_iterations = 12

        # check transition model for correct dimensions
        for i in range(len(state_shape)):
            assert np.shape(transition_model)[i] == state_shape[i]
        assert np.shape(transition_model)[len(state_shape)] == action_shape
        for i in range(len(state_shape)):
            assert np.shape(transition_model)[i + len(state_shape) + 1] == state_shape[i]
        self.transition_model = transition_model

        # check reward model for correct dimensions
        for i in range(len(state_shape)):
            assert np.shape(reward_model)[i] == state_shape[i]
        assert np.shape(reward_model)[len(state_shape)] == action_shape
        self.reward_model = reward_model

        # check observation model for correct dimensions
        for i in range(len(state_shape)):
            assert np.shape(observation_model)[i] == state_shape[i]
        assert np.shape(observation_model)[len(state_shape)] == self.num_obs
        self.observation_model = observation_model

        if b_initial is not None:
            for i in range(len(state_shape)):
                assert np.shape(b_initial)[i] == state_shape[i]
            self.belief = b_initial
        else:
            self.belief = np.ones(state_shape) / np.prod(state_shape)

        self.values = None
        self.__build_value_table()

        self.b_prime = self.belief

    def expert_action(self):
        """
        Select an action using the QMDP heuristic.
        :return: action
        """
        # compute Q(s,a)
        q = np.zeros(np.concatenate([self.state_shape, [self.action_shape]]))
        current_s_indices = []
        s_dims_to_go = self.state_shape
        self.__compute_q_helper(q, current_s_indices, s_dims_to_go)

        # compute Q_MDP(a)
        current_s_indices = []
        s_dims_to_go = self.state_shape
        q_mdp = self.__qmdp_helper(q, current_s_indices, s_dims_to_go)

        # choose action based on Q_MDP values
        a = np.argmax(q_mdp)
        return a

    def __compute_q_helper(self, q, current_s_indices, s_dims_to_go):
        # compute q values for all states matching current_s_indices
        if len(s_dims_to_go) == 0:
            # base case - compute q for this s
            for a_idx in range(self.action_shape):
                reward = self.reward_model[tuple(current_s_indices + [a_idx])]
                current_s1_indices = []
                s1_dims_to_go = self.state_shape
                next_value = self.__iterate_over_s1_helper(current_s_indices, current_s1_indices, s1_dims_to_go, a_idx)
                q[tuple(current_s_indices + [a_idx])] = reward + next_value
        else:
            # recursive case
            s0 = s_dims_to_go[0]
            s_dims_to_go = s_dims_to_go[1:]

            for i in range(s0):
                s_indices = current_s_indices + [i]
                self.__compute_q_helper(q, s_indices, s_dims_to_go)

    def __qmdp_helper(self, q, current_s_indices, s_dims_to_go):
        # sum b weighted q-values for all states matching current_s_indices
        if len(s_dims_to_go) == 0:
            # base case - return b weighted q values for each action for this s
            return q[tuple(current_s_indices)] * self.belief[tuple(current_s_indices)]  # vector of length |A|
        else:
            # recursive case
            s0 = s_dims_to_go[0]
            s_dims_to_go = s_dims_to_go[1:]

            return np.sum([self.__qmdp_helper(q, current_s_indices + [i], s_dims_to_go) for i in range(s0)], axis=0)

    def initialise_belief(self, obs):
        """
        Initialise the belief state based on an initial observation (from uniform prior)
        :param obs:
        """
        # update based on observation
        b_new = np.zeros(self.state_shape)
        o_idx = np.sum([obs[i] * np.prod([self.obs_shape[:i]]) for i in range(len(obs))])
        current_s_indices = []
        s_dims_to_go = self.state_shape
        self.__initialise_belief_helper(b_new, current_s_indices, s_dims_to_go, o_idx)

        # normalise belief
        self.belief = b_new / np.sum(b_new)

    def __initialise_belief_helper(self, b_new, current_s_indices, s_dims_to_go, o_idx):
        # initialise belief based on initial observation (from uniform prior) for all states matching current_s_indices
        if len(s_dims_to_go) == 0:
            # base case
            p_obs = self.observation_model[tuple(current_s_indices + [o_idx])]
            b_new[tuple(current_s_indices)] = p_obs
        else:
            # recursive case
            s0 = s_dims_to_go[0]
            s_dims_to_go = s_dims_to_go[1:]

            for i in range(s0):
                s_indices = current_s_indices + [i]
                self.__initialise_belief_helper(b_new, s_indices, s_dims_to_go, o_idx)

    def update_belief(self, a, obs):
        """
        Update the belief state using bayesian inference
        :param a:
        :param obs:
        """
        # update based on transition
        b_prime = np.zeros(self.state_shape)
        current_s_indices = []
        s_dims_to_go = self.state_shape
        self.__update_b_prime_s_helper(b_prime, current_s_indices, s_dims_to_go, a)

        # update based on observation
        b_new = np.zeros(self.state_shape)
        current_s_indices = []
        s_dims_to_go = self.state_shape
        o_idx = np.sum([obs[i] * np.prod([self.obs_shape[:i]]) for i in range(len(obs))])
        self.__update_b_obs_helper(b_new, b_prime, current_s_indices, s_dims_to_go, o_idx)

        # normalise belief
        self.belief = b_new / np.sum(b_new)

    def __update_b_prime_s_helper(self, b_prime, current_s_indices, s_dims_to_go, a_idx):
        # compute the new b' value for all states matching current_s_indices
        if len(s_dims_to_go) == 0:
            # base case - sum over all s1 for this s
            current_s1_indices = []
            s1_dims_to_go = self.state_shape
            b_new = self.__update_b_prime_s1_helper(current_s_indices, current_s1_indices, s1_dims_to_go, a_idx)
            b_prime[tuple(current_s_indices)] = b_new
        else:
            # recursive case
            s0 = s_dims_to_go[0]
            s_dims_to_go = s_dims_to_go[1:]

            for i in range(s0):
                s_indices = current_s_indices + [i]
                self.__update_b_prime_s_helper(b_prime, s_indices, s_dims_to_go, a_idx)

    def __update_b_prime_s1_helper(self, s_indices, current_s1_indices, s1_dims_to_go, a_idx):
        # compute the sum of partial beliefs for all previous states matching current s1 indices
        if len(s1_dims_to_go) == 0:
            # base case - partial belief from s1
            return self.transition_model[tuple(current_s1_indices + [a_idx] + s_indices)] * \
                   self.belief[tuple(current_s1_indices)]
        else:
            # recursive case
            s0 = s1_dims_to_go[0]
            s1_dims_to_go = s1_dims_to_go[1:]

            total = 0.0
            for i in range(s0):
                s1_indices = current_s1_indices + [i]
                total += self.__update_b_prime_s1_helper(s_indices, s1_indices, s1_dims_to_go, a_idx)

            return total

    def __update_b_obs_helper(self, b_new, b_prime, current_s_indices, s_dims_to_go, o_idx):
        # update belief for all states matching current_s_indices
        if len(s_dims_to_go) == 0:
            # base case - update belief based on obs for this s
            p_obs = self.observation_model[tuple(current_s_indices + [o_idx])]
            b_new[tuple(current_s_indices)] = b_prime[tuple(current_s_indices)] * p_obs
        else:
            # recursive case
            s0 = s_dims_to_go[0]
            s_dims_to_go = s_dims_to_go[1:]

            for i in range(s0):
                s_indices = current_s_indices + [i]
                self.__update_b_obs_helper(b_new, b_prime, s_indices, s_dims_to_go, o_idx)

    def force_belief_state(self, b):
        """
        Force the internal belief to be set to the given belief state
        :param b: new belief
        """
        self.belief = b

    def replan(self, transition_model=None, reward_model=None, observation_model=None, b_initial=None):
        """
        Replan with the given new transition model, reward model and observation model
        :param transition_model:
        :param reward_model:
        :param observation_model:
        :param b_initial:
        :return:
        """
        rebuild_value = False
        if transition_model is not None:
            # check for change
            if not np.array_equal(transition_model, self.transition_model):
                rebuild_value = True
                # check transition model for correct dimensions
                for i in range(len(self.state_shape)):
                    assert np.shape(transition_model)[i] == self.state_shape[i]
                assert np.shape(transition_model)[len(self.state_shape)] == self.action_shape
                for i in range(len(self.state_shape)):
                    assert np.shape(transition_model)[i + len(self.state_shape) + 1] == self.state_shape[i]
                self.transition_model = transition_model

        if reward_model is not None:
            # check for change
            if not np.array_equal(reward_model, self.reward_model):
                rebuild_value = True
                # check reward model for correct dimensions
                for i in range(len(self.state_shape)):
                    assert np.shape(reward_model)[i] == self.state_shape[i]
                assert np.shape(reward_model)[len(self.state_shape)] == self.action_shape
                self.reward_model = reward_model

        if observation_model is not None:
            # check for change
            if not np.array_equal(observation_model, self.observation_model):
                rebuild_value = True
                # check observation model for correct dimensions
                for i in range(len(self.state_shape)):
                    assert np.shape(observation_model)[i] == self.state_shape[i]
                assert np.shape(observation_model)[len(self.state_shape)] == self.num_obs
                self.observation_model = observation_model

        if rebuild_value:
            self.__build_value_table()

        if b_initial is not None:
            self.belief = b_initial

    def __update_belief_s_helper(self, current_s_indices, s_dims_to_go, a_idx):
        # recursive helper for updating belief for all states that match current indices
        if len(s_dims_to_go) == 0:
            # base case
            pass

        else:
            # recursive case
            s0 = s_dims_to_go[0]
            s_dims_to_go = s_dims_to_go[1:]

            for i in range(s0):
                s_indices = current_s_indices + [i]
                self.__update_belief_s_helper(s_indices, s_dims_to_go, a_idx)

    def __build_value_table(self):
        """
        Do value iteration.
        :return:
        """
        self.values = np.zeros(self.state_shape)
        for i in range(self.num_iterations):
            current_s_indices = []
            s_dims_to_go = self.state_shape
            actions = self.action_shape
            self.__iterate_over_s_helper(current_s_indices, s_dims_to_go, actions)

            # print("it " + str(i) + " complete")

    def __iterate_over_s_helper(self, current_s_indices, s_dims_to_go, actions):
        # recursive helper for building value table in n dimensions
        # return V(s) for current s

        if len(s_dims_to_go) == 0:
            # base case
            max_adv = 0.0
            # loop over a
            for a in range(actions):
                # compute advantage
                current_s1_indices = []
                s1_dims_to_go = self.state_shape

                reward = self.reward_model[tuple(current_s_indices + [a])]
                next_value = self.__iterate_over_s1_helper(current_s_indices, current_s1_indices, s1_dims_to_go, a)
                adv = reward + (self.GAMMA * next_value)
                if adv > max_adv:
                    max_adv = adv
            # update value for this state
            self.values[tuple(current_s_indices)] = max_adv

        else:
            # recursive case
            s0 = s_dims_to_go[0]
            s_dims_to_go = s_dims_to_go[1:]

            for i in range(1, s0 - 1):      # modified
                s_indices = current_s_indices + [i]
                self.__iterate_over_s_helper(s_indices, s_dims_to_go, actions)

    def __iterate_over_s1_helper(self, s_indices, current_s1_indices, s1_dims_to_go, a_idx):
        # compute expected value from s' for all s' matching current_s1_indices
        if len(s1_dims_to_go) == 0:
            # base case - return V(s') * T(s, a, s') for this s1
            s1_value = self.values[tuple(current_s1_indices)]
            t_model = self.transition_model[tuple(s_indices + [a_idx] + current_s1_indices)]
            return s1_value * t_model
        else:
            # recursive case - sum over all s1 which fit the current indices
            s0 = s1_dims_to_go[0]
            s1_dims_to_go = s1_dims_to_go[1:]
            total = 0.0
            for i in range(1, s0 - 1):      # modified
                s1_indices = current_s1_indices + [i]
                total += self.__iterate_over_s1_helper(s_indices, s1_indices, s1_dims_to_go, a_idx)

            return total






