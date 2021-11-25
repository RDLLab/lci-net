import numpy as np
import random
import time


class FullObsDynMazeV1:
    """
    Class representing a parameterised dynamic 2D maze environment.

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

    KERNEL_SIZE = 3

    # dynamic maze  specific parameters
    NUM_GATES = 2
    SWITCH_STEPS = 2

    def __init__(self, x_size, y_size, transition_prob, collision_penalty):
        assert x_size % 2 == 1
        self.x_size = x_size
        self.xk_max = (x_size - 1) // 2

        assert y_size % 2 == 1
        self.y_size = y_size
        self.yk_max = (y_size - 1) // 2

        self.t_prob = transition_prob
        self.collision_penalty = collision_penalty

        self.grid = None
        self.rewards = None

        self.agent_y = None
        self.agent_x = None
        self.goal_y = None
        self.goal_x = None

        self.p1_k = None
        self.p2_k = None
        self.initial_y = None
        self.initial_x = None

        self.steps = 0
        self.gates = None
        self.open_gate_idx = 0

        self.t_models = None
        self.o_models = None

        self.mutate_prob = 0.125

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
        return np.array([[2]])

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
        return [np.array([self.agent_y, self.agent_x])]

    def reset(self):
        # Generate a new "random" maze instance using randomised Prim's algorithm.
        np.random.seed(int(time.time()))

        # %%%%% build obstacle map %%%%%
        self.grid = np.ones([self.y_size, self.x_size], dtype=np.int8)

        in_maze = np.zeros([self.yk_max, self.xk_max], dtype=np.int8)
        wall_list = []

        # pick a random cell
        cell_xk = np.random.randint(self.xk_max)
        cell_yk = np.random.randint(self.yk_max)
        cell_x = (cell_xk * 2) + 1
        cell_y = (cell_yk * 2) + 1

        # TODO: modify this?
        starting_cell = (cell_yk, cell_xk)  # goal position should not be starting cell
        in_maze[cell_yk][cell_xk] = 1
        self.grid[cell_y][cell_x] = 0

        # add cell walls to wall list
        if cell_yk > 0 and self.grid[cell_y - 1][cell_x] == 1:
            wall_list.append((cell_y - 1, cell_x))
        if cell_yk < self.yk_max - 1 and self.grid[cell_y + 1][cell_x] == 1:
            wall_list.append((cell_y + 1, cell_x))
        if cell_xk > 0 and self.grid[cell_y][cell_x - 1] == 1:
            wall_list.append((cell_y, cell_x - 1))
        if cell_xk < self.xk_max - 1 and self.grid[cell_y][cell_x + 1] == 1:
            wall_list.append((cell_y, cell_x + 1))

        while len(wall_list) > 0:
            # pick a random wall from wall_list
            idx = np.random.choice(range(len(wall_list)), 1)[0]
            wall_y, wall_x = wall_list[idx]
            # determine if wall is horizontal or vertical
            if wall_y % 2 == 1:
                # horizontal wall
                cell_1_x = wall_x - 1
                cell_2_x = wall_x + 1
                cell_1_y = wall_y
                cell_2_y = wall_y
            else:
                # vertical wall
                cell_1_x = wall_x
                cell_2_x = wall_x
                cell_1_y = wall_y - 1
                cell_2_y = wall_y + 1
            # convert to k coordinates
            cell_1_xk = (cell_1_x - 1) // 2
            cell_2_xk = (cell_2_x - 1) // 2
            cell_1_yk = (cell_1_y - 1) // 2
            cell_2_yk = (cell_2_y - 1) // 2
            # test if strictly one cell is visited
            if in_maze[cell_1_yk][cell_1_xk] != in_maze[cell_2_yk][cell_2_xk]:
                # make the wall a passage
                self.grid[wall_y][wall_x] = 0
                # add unvisited cell to maze
                if in_maze[cell_1_yk][cell_1_xk]:
                    # cell 1 is in maze -> add cell 2 to maze
                    in_maze[cell_2_yk][cell_2_xk] = 1
                    self.grid[cell_2_y][cell_2_x] = 0

                    cell_x = cell_2_x
                    cell_y = cell_2_y
                    cell_xk = cell_2_xk
                    cell_yk = cell_2_yk
                else:
                    # cell 2 is in maze -> add cell 1 to maze
                    in_maze[cell_1_yk][cell_1_xk] = 1
                    self.grid[cell_1_y][cell_1_x] = 0

                    cell_x = cell_1_x
                    cell_y = cell_1_y
                    cell_xk = cell_1_xk
                    cell_yk = cell_1_yk
                # add neighbouring walls
                if cell_yk > 0 and self.grid[cell_y - 1][cell_x] == 1:
                    wall_list.append((cell_y - 1, cell_x))
                if cell_yk < self.yk_max - 1 and self.grid[cell_y + 1][cell_x] == 1:
                    wall_list.append((cell_y + 1, cell_x))
                if cell_xk > 0 and self.grid[cell_y][cell_x - 1] == 1:
                    wall_list.append((cell_y, cell_x - 1))
                if cell_xk < self.xk_max - 1 and self.grid[cell_y][cell_x + 1] == 1:
                    wall_list.append((cell_y, cell_x + 1))

            # remove wall from list
            wall_list.remove((wall_y, wall_x))

        # %%%%% generate dynamic gate positions %%%%%
        # divide into contiguous partitions and find boundary
        p1_k, p2_k, w_y, w_x = self.__partition_maze()
        boundary = self.__get_boundary(p1_k, p2_k)

        # self.debug_show_cells(self.grid, boundary)

        # select random boundary walls to become gates
        gate1 = (w_y, w_x)  # use the split point as one of the gates
        boundary.remove(gate1)
        gate2 = random.choice(boundary)
        self.gates = [gate1, gate2]
        self.open_gate_idx = 0

        # %%%%% generate initial and goal positions %%%%%
        # select initial position from partition 1
        y, x = random.choice(p1_k)
        self.agent_y, self.agent_x = (2 * y) + 1, (2 * x) + 1

        # select goal position from partition 2
        y, x = random.choice(p2_k)
        self.goal_y, self.goal_x = (2 * y) + 1, (2 * x) + 1

        # reset step count (to trigger next gate change)
        self.steps = 0

        # %%%%% build reward map %%%%%
        self.rewards = np.zeros([self.y_size, self.x_size], dtype=np.int8)
        # self.rewards = np.array(self.grid) * -1
        self.rewards[self.goal_y][self.goal_x] = 1

        # store initial position (for reset to start)
        self.p1_k = p1_k
        self.p2_k = p2_k
        self.initial_y = self.agent_y
        self.initial_x = self.agent_x

        # build MDP
        t_model = self.__build_transition_model()
        #o_model = self.__build_obs_model()
        self.mutate()
        t_model_mut = self.__build_transition_model()
        #o_model_mut = self.__build_obs_model()
        self.t_models = [t_model, t_model_mut]
        #self.o_models = [o_model, o_model_mut]

    def regen_start_and_goal_pos(self):
        """
        Re-generate starting and goal positions and move agent to new starting position
        """
        # reset gate status
        self.open_gate_idx = 0
        for gate in self.gates:
            if gate == self.gates[self.open_gate_idx]:
                self.grid[gate] = self.FREE_SPACE
            else:
                self.grid[gate] = self.OBSTACLE

        # re-seed random number generator
        random.seed(int(time.time()) + self.initial_y + (self.initial_x * 10))

        # swap p1 and p2
        temp = self.p1_k
        self.p1_k = self.p2_k
        self.p2_k = temp

        # select initial position from partition 1
        y, x = random.choice(self.p1_k)
        self.agent_y, self.agent_x = (2 * y) + 1, (2 * x) + 1

        # select goal position from partition 2
        y, x = random.choice(self.p2_k)
        self.goal_y, self.goal_x = (2 * y) + 1, (2 * x) + 1

        self.initial_y = self.agent_y
        self.initial_x = self.agent_x

        # build reward map
        self.rewards = np.zeros([self.y_size, self.x_size], dtype=np.int8)
        self.rewards[self.goal_y][self.goal_x] = 1

    def reset_to_start(self):
        """
        Reset agent to starting position (start position is not re-generated)
        """
        # reset gate status
        self.open_gate_idx = 0
        for gate in self.gates:
            if gate == self.gates[self.open_gate_idx]:
                self.grid[gate] = self.FREE_SPACE
            else:
                self.grid[gate] = self.OBSTACLE

        # reset agent position
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
                symbol = '   '
                # test for obstacle
                if self.grid[row][column]:
                    # render obstacle in this square
                    symbol = 'XXX'

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
            print(line)
        print(' ')
        print('-' * self.x_size * 3)
        print(' ')

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
        return belief

    def get_pomdp(self):
        """
        Return a POMDP description of the environment.
        :return: (T, R, Z)
        """
        return self.get_transition_model(), self.get_reward_model(), self.get_obs_model()

    def get_transition_model(self):
        """
        Return the currently active transition model.

        Format = [n, m, a, n', m']
        :return: transition model
        """
        return self.t_models[self.open_gate_idx]

    def get_reward_model(self):
        """
        Compute the underlying MDP reward model, R(s, a)

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
        Return the currently active observation model.

        Format = [n, m, o]
        :return: observation model
        """
        return None

    def __build_transition_model(self):
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

    def mutate(self):
        """
        Mutate the environment by switching which gate is open.

        If the open gate position is occupied by the agent the open gate will not be switched.

        Return True if mutation could be performed, False otherwise
        """
        # check that open gate is available
        gy, gx = self.gates[self.open_gate_idx]
        if not (gy == self.agent_y and gx == self.agent_x):
            # toggle open gate
            self.open_gate_idx = (self.open_gate_idx + 1) % self.NUM_GATES
            for gate in self.gates:
                if gate == self.gates[self.open_gate_idx]:
                    self.grid[gate] = self.FREE_SPACE
                else:
                    self.grid[gate] = self.OBSTACLE
            return True
        else:
            return False

    def get_random_b0(self):
        """
        Generate a random initial belief state.
        :return: initial belief state
        """
        return self.get_true_state()

    def __get_obs(self, y, x):
        return np.array([y, x])

    def __get_reward(self, y, x):
        """
        Reward defined as -1 * distance to goal
        :param y:
        :param x:
        :return:
        """
        # dist = math.sqrt(((self.agent_y - self.goal_y) ** 2) + ((self.agent_x - self.goal_x) ** 2))
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

    def __hold_pos(self, position):
        """
        Do not move.
        :param position: (ROW,COL)
        :return: (new ROW, new COL, collision [T/F])
        """
        r, c = position
        return r, c, False  # assumed to not be in collision

    # TODO: modify
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

    # TODO: modify
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

    # TODO: modify
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

    def __partition_maze(self):
        """

        :return: (partition 1, partition 2, split y, split x)
        """
        grid_clone = np.array(self.grid)

        # %%%%% choose a random open wall and turn it into a closed wall %%%%%
        while True:
            # pick a random wall cell
            w_y = np.random.randint(1, self.y_size - 1)
            if w_y % 2 == 1:
                # horizontal wall
                w_xk = np.random.randint(self.xk_max)
                w_x = (2 * w_xk) + 2
            else:
                # vertical wall
                w_xk = np.random.randint(self.xk_max)
                w_x = (2 * w_xk) + 1
            # check if wall is open
            if self.grid[w_y][w_x] == self.FREE_SPACE:
                break
        # turn open wall into a closed wall
        grid_clone[w_y][w_x] = self.OBSTACLE

        # %%%%% build the two partitions formed by the newly closed wall
        if w_y % 2 == 1:
            # horizontal wall
            c1_y = w_y
            c1_x = w_x - 1
            c2_y = w_y
            c2_x = w_x + 1
        else:
            # vertical wall
            c1_y = w_y - 1
            c1_x = w_x
            c2_y = w_y + 1
            c2_x = w_x
        c1_yk = (c1_y - 1) // 2
        c1_xk = (c1_x - 1) // 2
        c2_yk = (c2_y - 1) // 2
        c2_xk = (c2_x - 1) // 2

        p1_k = self.__get_cts_cells(grid_clone, c1_yk, c1_xk)
        p2_k = self.__get_cts_cells(grid_clone, c2_yk, c2_xk)

        return p1_k, p2_k, w_y, w_x

    def __get_cts_cells(self, grid, c_yk, c_xk):
        visited = [(c_yk, c_xk)]
        to_expand = [(c_yk, c_xk)]
        while len(to_expand) > 0:
            yk, xk = to_expand.pop(0)
            # add cell above
            if grid[yk * 2][(xk * 2) + 1] == self.FREE_SPACE and (yk - 1, xk) not in visited:
                to_expand.append((yk - 1, xk))
                visited.append((yk - 1, xk))
            # add cell below
            if grid[(yk * 2) + 2][(xk * 2) + 1] == self.FREE_SPACE and (yk + 1, xk) not in visited:
                to_expand.append((yk + 1, xk))
                visited.append((yk + 1, xk))
            # add cell to left
            if grid[(yk * 2) + 1][xk * 2] == self.FREE_SPACE and (yk, xk - 1) not in visited:
                to_expand.append((yk, xk - 1))
                visited.append((yk, xk - 1))
            # add cell to right
            if grid[(yk * 2) + 1][(xk * 2) + 2] == self.FREE_SPACE and (yk, xk + 1) not in visited:
                to_expand.append((yk, xk + 1))
                visited.append((yk, xk + 1))
        return visited

    @staticmethod
    def __get_boundary(p1_k, p2_k):
        boundary = []
        for y1_k, x1_k in p1_k:
            # check above
            if (y1_k - 1, x1_k) in p2_k and ((y1_k * 2) + 0, (x1_k * 2) + 1) not in boundary:
                boundary.append(((y1_k * 2) + 0, (x1_k * 2) + 1))
            # check below
            if (y1_k + 1, x1_k) in p2_k and ((y1_k * 2) + 2, (x1_k * 2) + 1) not in boundary:
                boundary.append(((y1_k * 2) + 2, (x1_k * 2) + 1))
            # check left
            if (y1_k, x1_k - 1) in p2_k and ((y1_k * 2) + 1, (x1_k * 2) + 0) not in boundary:
                boundary.append(((y1_k * 2) + 1, (x1_k * 2) + 0))
            # check right
            if (y1_k, x1_k + 1) in p2_k and ((y1_k * 2) + 1, (x1_k * 2) + 2) not in boundary:
                boundary.append(((y1_k * 2) + 1, (x1_k * 2) + 2))
        return boundary

    def debug_show_cells(self, grid, cells):
        for row in range(self.y_size):
            line = ''
            for column in range(self.x_size):
                symbol = '   '
                # test for obstacle
                if grid[row][column]:
                    # render obstacle in this square
                    symbol = '\u2588\u2588\u2588'

                if (row, column) in cells:
                    symbol = '\u2588B\u2588'

                line += symbol
            print(line)
        print(' ')
        print('-' * self.x_size * 3)
        print(' ')


def combination(n, r):
    return np.math.factorial(n) / (np.math.factorial(n - r) * np.math.factorial(r))

