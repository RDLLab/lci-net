import numpy as np
import random
import time
import PIL
from PIL import Image


class ImageGrid2D:
    """
    Class representing a single 2D grid world environment derived from a black
    and white bitmap image.

    This class is an environment.
    """

    # matrix indices
    ROW = 0
    COL = 1

    # action set
    HOLD_POS = 4
    NORTH = 3
    SOUTH = 1
    EAST = 0
    WEST = 2

    NUM_ACTIONS = 5

    OBSTACLE = 1
    FREE_SPACE = 0

    MIN_PATH_LENGTH = 5

    KERNEL_SIZE = 3

    def __init__(self, filename, obs_dirs, transition_prob,
                 obs_success_prob, collision_penalty):
        self.obs_dirs = obs_dirs    # list of tuples, each giving a relative position
        self.t_prob = transition_prob
        self.obs_success_prob = obs_success_prob
        self.collision_penalty = collision_penalty

        self.grid = None
        self.rewards = None

        self.agent_y = None
        self.agent_x = None
        self.goal_y = None
        self.goal_x = None

        self.initial_y = None
        self.initial_x = None

        self.mutate_prob = 0

        # load obstacle map from a bmp image
        img = PIL.Image.open(filename)
        self.x_size, self.y_size = img.size

        self.grid = np.zeros((self.y_size, self.x_size))
        for r in range(self.y_size):
            for c in range(self.x_size):
                intensity = img.getpixel((c, r))[0]
                if intensity < 127:
                    self.grid[r][c] = 1

        self.reset()

    def get_state_tensor_shape(self):
        """
        Return shape of state space, which here is the dimensions of the grid world
        :return: State tensor shape
        """
        return np.array([self.y_size, self.x_size])

    def get_actions_tensor_shape(self):
        """
        Return shape of action space. Here there are 5 actions (for one agent only).
        :return: Action tensor shape
        """
        return np.array([self.NUM_ACTIONS])

    def get_obs_tensor_shape(self):
        """
        Return shape of observation space.
        :return: Observation tensor shape
        """
        return np.array([[2 for i in self.obs_dirs]])

    def get_transition_tensor_shape(self):
        """
        Return the shape of the local transition kernel.

        Note: the agent is responsible for filling out the values of the kernel
        :return: Transition kernel shape
        """
        return np.array([self.KERNEL_SIZE, self.KERNEL_SIZE])

    def get_map_param_tensor_shape(self):
        """
        Return the shape of the map parameters. Here, the map parameters
        represent an obstacle map.
        :return: shape of theta_m
        """
        return np.array(np.shape(self.grid))

    def get_task_param_tensor_shape(self):
        """
        Return the shape of the task parameters. Here, the task parameters
        represent a map with the goal position marked as 1, all other positions
        marked as 0.
        :return: shape of theta_t
        """
        return np.array(np.shape(self.rewards))

    def get_map_params(self):
        """
        :return: map params
        """
        return self.grid

    def get_task_params(self):
        """
        :return: task params
        """
        return self.rewards

    def get_init_obs(self):
        """
        Return the initial observation received before the agent takes any action
        :return: initial observation
        """
        return [self.__get_obs(self.agent_y, self.agent_x)]

    def reset(self):
        self.regen_start_and_goal_pos()

    def regen_start_and_goal_pos(self):
        """
        Re-generate starting and goal positions and move agent to new starting position
        """
        # %%%%% generate initial and goal positions %%%%%
        # re-seed random number generator
        # np.random.seed(int(time.time()) + self.initial_y + (self.initial_x * 10))
        np.random.seed(int(time.time()))
        # loop until valid combination found
        while True:
            # loop until collision free initial position is found
            while True:
                self.agent_y = np.random.randint(self.y_size)
                self.agent_x = np.random.randint(self.x_size)
                if self.grid[self.agent_y][self.agent_x] == self.FREE_SPACE:
                    # no obstacle
                    break

            # loop until collision free goal position is found
            while True:
                self.goal_y = np.random.randint(self.y_size)
                self.goal_x = np.random.randint(self.x_size)
                if self.grid[self.goal_y][self.goal_x] == self.FREE_SPACE:
                    # no obstacle
                    break

            # check for path between initial and goal
            if (self.__path_exists(self.agent_y, self.agent_x, self.goal_y, self.goal_x) and
                    get_manhattan_dist(self.agent_y, self.agent_x, self.goal_y, self.goal_x) >= self.MIN_PATH_LENGTH):
                # valid initial and goal combination
                break
        # build reward map
        self.rewards = np.zeros([self.y_size, self.x_size], dtype=np.int8)
        self.rewards[self.goal_y][self.goal_x] = 1

        # store initial positions (for reset to start)
        self.initial_y = self.agent_y
        self.initial_x = self.agent_x

    def reset_to_start(self):
        """
        Reset agent to starting position (start position is not re-generated)
        """
        self.agent_y = self.initial_y
        self.agent_x = self.initial_x

    def step(self, actions):
        """
        Increment the simulation by 1 step based on the given joint action and
        return the rewards and observations received by each of the agents (as
        well as a 'done' flag).
        :param actions: [a1, a2, ... aN]
        :return: ([r1, r2, ... rN], [o1, o2, ... oN], done)
        """

        a = actions[0]
        # move in the chosen direction with probability p, random other direction with probability 1-p
        if random.random() < self.t_prob:
            # move in the chosen direction
            direction = a
        else:
            # sample a non-chosen direction at uniform random
            dirs = [self.HOLD_POS, self.NORTH, self.SOUTH, self.EAST, self.WEST]
            dirs.remove(a)
            direction = random.choice(dirs)

        # execute move
        switcher = {self.HOLD_POS: self.__hold_pos, self.NORTH: self.__move_north, self.SOUTH: self.__move_south,
                    self.EAST: self.__move_east, self.WEST: self.__move_west}
        move = switcher.get(direction)
        row, column, collision = move((self.agent_y, self.agent_x))

        # update internal state
        self.agent_y = row
        self.agent_x = column

        reward = self.__get_reward(row, column) - (collision * self.collision_penalty)
        # if a == self.HOLD_POS:
        #     reward -= 5
        rewards = [reward]

        observation = self.__get_obs(row, column)
        observations = [observation]

        done = self.__is_terminal()

        return rewards, observations, done

    def render(self):
        """
        Output a human readable text representation of the environment.
        """
        for row in range(self.y_size):
            line = ''
            for column in range(self.x_size):
                # line += '['
                symbol = ' '
                # test for obstacle
                if self.grid[row][column]:
                    # render obstacle in this square
                    symbol = 'X'

                # test for goal
                if self.goal_y == row and self.goal_x == column:
                    # render goal in this square
                    symbol = 'G'

                # test for agent
                if self.agent_y == row and self.agent_x == column:
                    # render agent in this square
                    symbol = 'A'

                line += symbol
                # line += ']'
            print(line)
        # print('-' * self.x_size * 3)

    def render_to_file(self, file):
        """
        Write a human readable text representation of the environment to file.
        """
        for row in range(self.y_size):
            line = ''
            for column in range(self.x_size):
                # line += '['
                symbol = '   '
                # test for obstacle
                if self.grid[row][column]:
                    # render obstacle in this square
                    symbol = '\u2588\u2588\u2588'

                # test for goal
                if self.goal_y == row and self.goal_x == column:
                    # render goal in this square
                    symbol = ' G '

                # test for agent
                if self.agent_y == row and self.agent_x == column:
                    # render agent in this square
                    symbol = ' A '

                line += symbol
                # line += ']'
            file.write(line + "\n")

    def get_true_state(self):
        """
        Output a belief distribution with probability of 1 for the true state.
        :return: true state belief distribution
        """
        belief = np.zeros([self.y_size, self.x_size])
        belief[self.agent_y][self.agent_x] = 1
        return np.array([belief])

    def get_pomdp(self):
        """
        Return a POMDP description of the environment.
        :return: (T, R, Z)
        """
        return self.get_transition_model(), self.get_reward_model(), self.get_obs_model()

    def get_transition_model(self):
        """
        Compute underlying MDP transition model, T(s,a,s')

        Format = [n, m, a, n', m']
        :return: transition model
        """
        transition_model = np.zeros([self.y_size, self.x_size, self.NUM_ACTIONS, self.y_size, self.x_size])
        # loop across all non border cells
        # border cells can never be occupied, so don't care about these values
        for r in range(1, self.y_size - 1):
            for c in range(1, self.x_size - 1):
                # if cell is obstacle, don't need to compute
                if self.grid[r][c] == self.OBSTACLE:
                    continue

                surplus_hold = 0
                surplus_north = 0
                surplus_south = 0
                surplus_east = 0
                surplus_west = 0

                # transition probabilities for cell to north
                if self.grid[r - 1][c] == self.FREE_SPACE:
                    # cell to north is free
                    transition_model[r][c][self.HOLD_POS][r - 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    transition_model[r][c][self.NORTH][r - 1][c] = self.t_prob
                    transition_model[r][c][self.SOUTH][r - 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    transition_model[r][c][self.EAST][r - 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    transition_model[r][c][self.WEST][r - 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                else:
                    # cell to north is obstacle - add surplus probability for all actions
                    transition_model[r][c][self.HOLD_POS][r - 1][c] = 0
                    transition_model[r][c][self.NORTH][r - 1][c] = 0
                    transition_model[r][c][self.SOUTH][r - 1][c] = 0
                    transition_model[r][c][self.EAST][r - 1][c] = 0
                    transition_model[r][c][self.WEST][r - 1][c] = 0
                    surplus_hold += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    surplus_north += self.t_prob
                    surplus_south += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    surplus_east += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    surplus_west += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)

                # transition probabilities for cell to south
                if self.grid[r + 1][c] == self.FREE_SPACE:
                    # cell to south is free
                    transition_model[r][c][self.HOLD_POS][r + 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    transition_model[r][c][self.NORTH][r + 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    transition_model[r][c][self.SOUTH][r + 1][c] = self.t_prob
                    transition_model[r][c][self.EAST][r + 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    transition_model[r][c][self.WEST][r + 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                else:
                    # cell to south is obstacle - add surplus probability for all actions
                    transition_model[r][c][self.HOLD_POS][r + 1][c] = 0
                    transition_model[r][c][self.NORTH][r + 1][c] = 0
                    transition_model[r][c][self.SOUTH][r + 1][c] = 0
                    transition_model[r][c][self.EAST][r + 1][c] = 0
                    transition_model[r][c][self.WEST][r + 1][c] = 0
                    surplus_hold += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    surplus_north += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    surplus_south += self.t_prob
                    surplus_east += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    surplus_west += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)

                # transition probabilities for cell to east
                if self.grid[r][c + 1] == self.FREE_SPACE:
                    # cell to east is free
                    transition_model[r][c][self.HOLD_POS][r][c + 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    transition_model[r][c][self.NORTH][r][c + 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    transition_model[r][c][self.SOUTH][r][c + 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    transition_model[r][c][self.EAST][r][c + 1] = self.t_prob
                    transition_model[r][c][self.WEST][r][c + 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                else:
                    # cell to east is obstacle - add surplus probability for all actions
                    transition_model[r][c][self.HOLD_POS][r][c + 1] = 0
                    transition_model[r][c][self.NORTH][r][c + 1] = 0
                    transition_model[r][c][self.SOUTH][r][c + 1] = 0
                    transition_model[r][c][self.EAST][r][c + 1] = 0
                    transition_model[r][c][self.WEST][r][c + 1] = 0
                    surplus_hold += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    surplus_north += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    surplus_south += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    surplus_east += self.t_prob
                    surplus_west += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)

                # transition probabilities for cell to west
                if self.grid[r][c - 1] == self.FREE_SPACE:
                    # cell to west is free
                    transition_model[r][c][self.HOLD_POS][r][c - 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    transition_model[r][c][self.NORTH][r][c - 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    transition_model[r][c][self.SOUTH][r][c - 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    transition_model[r][c][self.EAST][r][c - 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    transition_model[r][c][self.WEST][r][c - 1] = self.t_prob
                else:
                    # cell to west is obstacle - add surplus probability for all actions
                    transition_model[r][c][self.HOLD_POS][r][c - 1] = 0
                    transition_model[r][c][self.NORTH][r][c - 1] = 0
                    transition_model[r][c][self.SOUTH][r][c - 1] = 0
                    transition_model[r][c][self.EAST][r][c - 1] = 0
                    transition_model[r][c][self.WEST][r][c - 1] = 0
                    surplus_hold += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    surplus_north += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    surplus_south += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    surplus_east += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    surplus_west += self.t_prob

                # transition probabilities for staying in the same cell
                transition_model[r][c][self.HOLD_POS][r][c] = self.t_prob + surplus_hold
                transition_model[r][c][self.NORTH][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1) + surplus_north
                transition_model[r][c][self.SOUTH][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1) + surplus_south
                transition_model[r][c][self.EAST][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1) + surplus_east
                transition_model[r][c][self.WEST][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1) + surplus_west

        return transition_model

    def get_reward_model(self):
        """
        Compute the underlying MDP reward model, R(s,a)

        Format = [n, m, a]
        :return: reward model
        """
        reward_model = np.ones([self.y_size, self.x_size, self.NUM_ACTIONS]) * -1

        # assign reward of +1 for hold position action in goal cell
        reward_model[self.goal_y][self.goal_x][self.HOLD_POS] = 1

        # assign collision penalties for all colliding actions
        for r in range(1, self.y_size - 1):
            for c in range(1, self.x_size - 1):
                # north direction
                if self.grid[r - 1][c] == self.OBSTACLE:
                    reward_model[r][c][self.NORTH] -= self.collision_penalty
                # south direction
                if self.grid[r + 1][c] == self.OBSTACLE:
                    reward_model[r][c][self.SOUTH] -= self.collision_penalty
                # east direction
                if self.grid[r][c + 1] == self.OBSTACLE:
                    reward_model[r][c][self.EAST] -= self.collision_penalty
                # west direction
                if self.grid[r][c - 1] == self.OBSTACLE:
                    reward_model[r][c][self.WEST] -= self.collision_penalty

        # assign obstacles reward of zero
        for r in range(self.y_size):
            for c in range(self.x_size):
                if self.grid[r][c] == self.OBSTACLE:
                    for a in range(self.NUM_ACTIONS):
                        reward_model[r][c][a] = 0

        return reward_model

    def get_obs_model(self):
        """
        Compute the underlying MDP observation model, Z(o|s).

        Format = [n, m, o]
        :return: observation model
        """
        num_dirs = len(self.obs_dirs)
        num_obs = np.power(2, len(self.obs_dirs))
        # enumerate observations
        observations = []
        for o_right in [self.FREE_SPACE, self.OBSTACLE]:
            for o_down in [self.FREE_SPACE, self.OBSTACLE]:
                for o_left in [self.FREE_SPACE, self.OBSTACLE]:
                    for o_up in [self.FREE_SPACE, self.OBSTACLE]:
                        observations.append([o_up, o_left, o_down, o_right])

        # enumerate probabilities of receiving 0, 1, ... num_dirs correct observations
        p_correct = [combination(num_dirs, i) *
                     np.power(self.obs_success_prob, i) *
                     np.power(1 - self.obs_success_prob, num_dirs - i) for i in range(num_dirs + 1)]

        obs_model = np.zeros([self.y_size, self.x_size, num_obs])

        # loop over all states (except borders)
        for r in range(1, self.y_size - 1):
            for c in range(1, self.x_size - 1):
                # if cell is obstacle, Z(o|s) = 0 for all o
                if self.grid[r][c] == self.OBSTACLE:
                    for o in range(num_obs):
                        obs_model[r][c][o] = 0.0
                else:
                    # loop over all observations
                    for o in range(num_obs):
                        # find the number of directions for which the state is consistent with the observation
                        num_correct = 0
                        for i in range(len(self.obs_dirs)):
                            direction = self.obs_dirs[i]
                            r1 = r + direction[0]
                            c1 = c + direction[1]
                            if self.grid[r1][c1] == observations[o][i]:
                                # state is consistent with observation in this direction
                                num_correct += 1
                        obs_model[r][c][o] = p_correct[num_correct]

        return obs_model

    def mutate(self):
        """
        This environment is static - do nothing
        """
        return False

    def get_random_b0(self):
        """
        Generate a random initial belief state.
        :return: initial belief state
        """
        grid_temp = np.array(self.grid)
        grid_temp[self.initial_y][self.initial_x] = self.OBSTACLE   # don't include initial state
        free_states = np.nonzero((grid_temp == self.FREE_SPACE).flatten())[0]
        freespace_size = len(free_states)

        b0sizes = np.floor(freespace_size / np.power(2.0, np.arange(20)))
        b0sizes = b0sizes[:np.nonzero(b0sizes < 1)[0][0]]
        b0size = int(np.random.choice(b0sizes))

        b0ind = np.random.choice(free_states, b0size, replace=False)
        b0 = np.zeros([self.y_size * self.x_size])
        b0[b0ind] = 1.0 / (b0size + 1)  # uniform distribution over sampled states

        b0 = np.reshape(b0, [self.y_size, self.x_size])
        b0[self.initial_y][self.initial_x] = 1.0 / (b0size + 1)     # add initial state to initial belief set
        return b0

    def __path_exists(self, y1, x1, y2, x2):
        # path guaranteed to exist for included images
        return True

    def __get_obs(self, y, x):
        obs = []
        for dy, dx in self.obs_dirs:
            if random.random() < self.obs_success_prob:
                # correct observation
                o = self.grid[y + dy][x + dx]
            else:
                # incorrect observation
                o = 1 - self.grid[y + dy][x + dx]
            obs.append(o)
        return np.array(obs)

    def __get_reward(self, y, x):
        """
        Reward defined as -1 * distance to goal
        :param y:
        :param x:
        :return:
        """
        # dist = math.sqrt(((y - self.goal_y) ** 2) + ((x - self.goal_x) ** 2))
        # return -1 * dist

        if y == self.goal_y and x == self.goal_x:
            return 1.0
        else:
            return 0.0

    def __is_terminal(self):
        if self.agent_y == self.goal_y and self.agent_x == self.goal_x:
            return True
        else:
            return False

    @staticmethod
    def __hold_pos(position):
        """
        Do not move.
        :param position: (ROW,COL)
        :return: (new ROW, new COL, collision [T/F])
        """
        r, c = position
        return r, c, False  # assumed to not be in collision

    def __move_north(self, position):
        """
        Attempt to move the agent with the given position in the North direction.
        :param position: (ROW,COL)
        :return: (new ROW, new COL, collision [T/F])
        """
        r, c = position

        # check for out of bounds
        if r == 0:
            # can't move without hitting boundary
            return r, c, True

        # check for obstacle
        if self.grid[r - 1][c]:
            # can't move without hitting obstacle
            return r, c, True

        # move is possible without collision
        return r - 1, c, False

    def __move_south(self, position):
        """
        Attempt to move the agent with the given position in the South direction.
        :param position: (ROW,COL)
        :return: (new ROW, new COL, collision [T/F])
        """
        r, c = position

        # check for out of bounds
        if r == self.y_size - 1:
            # can't move without hitting boundary
            return r, c, True

        # check for obstacle
        if self.grid[r + 1][c]:
            # can't move without hitting obstacle
            return r, c, True

        # move is possible without collision
        return r + 1, c, False

    def __move_east(self, position):
        """
        Attempt to move the agent with the given position in the East direction.
        :param position: (ROW,COL)
        :return: (new ROW, new COL, collision [T/F])
        """
        r, c = position

        # check for out of bounds
        if c == self.x_size - 1:
            # can't move without hitting boundary
            return r, c, True

        # check for obstacle
        if self.grid[r][c + 1]:
            # can't move without hitting obstacle
            return r, c, True

        # move is possible without collision
        return r, c + 1, False

    def __move_west(self, position):
        """
        Attempt to move the agent with the given position in the West direction.
        :param position: (ROW,COL)
        :return: (new ROW, new COL, collision [T/F])
        """
        r, c = position

        # check for out of bounds
        if c == 0:
            # can't move without hitting boundary
            return r, c, True

        # check for obstacle
        if self.grid[r][c - 1]:
            # can't move without hitting obstacle
            return r, c, True

        # move is possible without collision
        return r, c - 1, False


def get_manhattan_dist(y1, x1, y2, x2):
    return abs(y2 - y1) + abs(x2 - x1)


def combination(n, r):
    return np.math.factorial(n) / (np.math.factorial(n - r) * np.math.factorial(r))

