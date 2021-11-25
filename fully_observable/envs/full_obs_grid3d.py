import numpy as np
import random
import time


class FullObsGrid3D:
    """
    Class representing a parameterised 3D grid world environment (e.d. for quad-rotor drone navigation).

    This class is an environment.
    """

    # matrix indices
    ROW = 0
    COL = 1
    PG = 2

    # action set
    HOLD_POS = 6
    NORTH = 3
    SOUTH = 1
    EAST = 0
    WEST = 2
    ASCEND = 4
    DESCEND = 5

    NUM_ACTIONS = 7

    OBSTACLE = 1
    FREE_SPACE = 0

    MIN_PATH_LENGTH = 0

    KERNEL_SIZE = 3

    def __init__(self, x_size, y_size, z_size, obstacle_prob, transition_prob, collision_penalty):

        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.obstacle_prob = obstacle_prob
        self.t_prob = transition_prob
        self.collision_penalty = collision_penalty

        self.grid = None
        self.rewards = None

        self.agent_z = None
        self.agent_y = None
        self.agent_x = None

        self.goal_z = None
        self.goal_y = None
        self.goal_x = None

        self.initial_z = None
        self.initial_y = None
        self.initial_x = None

        self.mutate_prob = 0

        self.reset()

    def get_state_tensor_shape(self):
        """
        Return shape of state space, which here is the dimensions of the grid world
        :return: State tensor shape
        """
        return np.array([self.z_size, self.y_size, self.x_size])

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
        return np.array([[self.z_size, self.y_size, self.x_size]])

    def get_transition_tensor_shape(self):
        """
        Return the shape of the local transition kernel.

        Note: the agent is responsible for filling out the values of the kernel
        :return: Transition kernel shape
        """
        return np.array([self.KERNEL_SIZE, self.KERNEL_SIZE, self.KERNEL_SIZE])

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
        return [np.array([self.agent_z, self.agent_y, self.agent_x])]

    def reset(self):
        # generate a new "random" grid instance
        np.random.seed(int(time.time()))

        # %%%%% build obstacle map %%%%%
        self.grid = np.zeros([self.z_size, self.y_size, self.x_size], dtype=np.int8)
        # borders
        self.grid[0, :, :] = self.OBSTACLE
        self.grid[-1, :, :] = self.OBSTACLE
        self.grid[:, 0, :] = self.OBSTACLE
        self.grid[:, -1, :] = self.OBSTACLE
        self.grid[:, :, 0] = self.OBSTACLE
        self.grid[:, :, -1] = self.OBSTACLE
        # add obstacles
        rand_field = np.random.rand(self.z_size, self.y_size, self.x_size)
        self.grid = np.array(np.logical_or(self.grid, (rand_field < self.obstacle_prob)), 'i')

        # find partitions
        disconnected_cells = []
        for p in range(self.z_size):
            for r in range(self.y_size):
                for c in range(self.x_size):
                    if self.grid[p][r][c] == self.FREE_SPACE:
                        disconnected_cells.append((p, r, c))
        partition_list = []
        while len(disconnected_cells) > 0:
            p, r, c = disconnected_cells[0]
            partition = []
            neighbours = [(p, r, c)]
            while len(neighbours) > 0:
                p, r, c = neighbours.pop(0)
                partition.append((p, r, c))
                disconnected_cells.remove((p, r, c))
                if (p-1, r, c) in disconnected_cells and (p-1, r, c) not in partition and (p-1, r, c) not in neighbours:
                    neighbours.append((p-1, r, c))
                if (p+1, r, c) in disconnected_cells and (p+1, r, c) not in partition and (p+1, r, c) not in neighbours:
                    neighbours.append((p+1, r, c))
                if (p, r-1, c) in disconnected_cells and (p, r-1, c) not in partition and (p, r-1, c) not in neighbours:
                    neighbours.append((p, r-1, c))
                if (p, r+1, c) in disconnected_cells and (p, r+1, c) not in partition and (p, r+1, c) not in neighbours:
                    neighbours.append((p, r+1, c))
                if (p, r, c-1) in disconnected_cells and (p, r, c-1) not in partition and (p, r, c-1) not in neighbours:
                    neighbours.append((p, r, c-1))
                if (p, r, c+1) in disconnected_cells and (p, r, c+1) not in partition and (p, r, c+1) not in neighbours:
                    neighbours.append((p, r, c+1))
            # all neighbours added to partition
            partition_list.append(partition)

        # merge partitions
        attempts = 0
        while len(partition_list) > 1:
            p1 = partition_list.pop(0)
            p2 = partition_list.pop(0)
            # find the set of cells on the boundary between p1 and p2
            boundary = set()
            for z1, y1, x1 in p1:
                # check above
                if (z1 - 2, y1, x1) in p2:
                    boundary.add((z1 - 1, y1, x1))
                # check below
                if (z1 + 2, y1, x1) in p2:
                    boundary.add((z1 + 1, y1, x1))
                # check north
                if (z1, y1 - 2, x1) in p2:
                    boundary.add((z1, y1 - 1, x1))
                # check south
                if (z1, y1 + 2, x1) in p2:
                    boundary.add((z1, y1 + 1, x1))
                # check west
                if (z1, y1, x1 - 2) in p2:
                    boundary.add((z1, y1, x1 - 1))
                # check east
                if (z1, y1, x1 + 2) in p2:
                    boundary.add((z1, y1, x1 + 1))
            if len(boundary) > 0:
                # delete a random wall between the partitions
                z_wall, y_wall, x_wall = random.sample(boundary, 1)[0]
                self.grid[z_wall, y_wall, x_wall] = self.FREE_SPACE
                # add new combined partition back into partition list
                partition_list.append(p1 + p2)
            else:
                # separated by more than one wall - shuffle partitions and try again
                partition_list.append(p1)
                partition_list.append(p2)
                random.shuffle(partition_list)
                attempts += 1

                if attempts > 20:
                    print("(!) failed to merge partitions")
                    break

        # %%%%% generate initial and goal positions %%%%%
        # re-seed random number generator
        np.random.seed(int(time.time()))
        # loop until valid combination found
        while True:
            # loop until collision free initial position is found
            while True:
                self.agent_z = np.random.randint(self.z_size)
                self.agent_y = np.random.randint(self.y_size)
                self.agent_x = np.random.randint(self.x_size)
                if self.grid[self.agent_z][self.agent_y][self.agent_x] == self.FREE_SPACE:
                    # no obstacle
                    break

            # loop until collision free goal position is found
            while True:
                self.goal_z = np.random.randint(self.z_size)
                self.goal_y = np.random.randint(self.y_size)
                self.goal_x = np.random.randint(self.x_size)
                if self.grid[self.goal_z][self.goal_y][self.goal_x] == self.FREE_SPACE:
                    # no obstacle
                    break

            # check for path between initial and goal
            # if self.__path_len(self.agent_z, self.agent_y, self.agent_x, self.goal_z, self.goal_y, self.goal_x) \
            #         >= self.MIN_PATH_LENGTH:
            #     # valid initial and goal combination
            #     break
            break

        # build reward map
        self.rewards = np.zeros([self.z_size, self.y_size, self.x_size], dtype=np.int8)
        self.rewards[self.goal_z][self.goal_y][self.goal_x] = 1

        # store initial positions (for reset to start)
        self.initial_z = self.agent_z
        self.initial_y = self.agent_y
        self.initial_x = self.agent_x

    def regen_start_and_goal_pos(self):
        """
        Re-generate starting and goal positions and move agent to new starting position
        """
        # %%%%% generate initial and goal positions %%%%%
        # re-seed random number generator
        np.random.seed(int(time.time()) + self.initial_y + (self.initial_x * 10))
        # loop until valid combination found
        while True:
            # loop until collision free initial position is found
            while True:
                self.agent_z = np.random.randint(self.z_size)
                self.agent_y = np.random.randint(self.y_size)
                self.agent_x = np.random.randint(self.x_size)
                if self.grid[self.agent_z][self.agent_y][self.agent_x] == self.FREE_SPACE:
                    # no obstacle
                    break

            # loop until collision free goal position is found
            while True:
                self.goal_z = np.random.randint(self.z_size)
                self.goal_y = np.random.randint(self.y_size)
                self.goal_x = np.random.randint(self.x_size)
                if self.grid[self.goal_z][self.goal_y][self.goal_x] == self.FREE_SPACE:
                    # no obstacle
                    break

            # check for path between initial and goal
            # if self.__path_len(self.agent_z, self.agent_y, self.agent_x, self.goal_z, self.goal_y, self.goal_x) \
            #         >= self.MIN_PATH_LENGTH:
            #     # valid initial and goal combination
            #     break
            break

        # build reward map
        self.rewards = np.zeros([self.z_size, self.y_size, self.x_size], dtype=np.int8)
        self.rewards[self.goal_z][self.goal_y][self.goal_x] = 1

        # store initial positions (for reset to start)
        self.initial_z = self.agent_z
        self.initial_y = self.agent_y
        self.initial_x = self.agent_x

    def reset_to_start(self):
        """
        Reset agent to starting position (start position is not re-generated)
        """
        self.agent_z = self.initial_z
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
            dirs = [self.HOLD_POS, self.NORTH, self.SOUTH, self.EAST, self.WEST, self.ASCEND, self.DESCEND]
            dirs.remove(a)
            direction = random.choice(dirs)

        # execute move
        switcher = {self.HOLD_POS: self.__hold_pos,
                    self.NORTH: self.__move_north, self.SOUTH: self.__move_south,
                    self.EAST: self.__move_east, self.WEST: self.__move_west,
                    self.ASCEND: self.__move_up, self.DESCEND: self.__move_down}
        move = switcher.get(direction)
        page, row, column, collision = move((self.agent_z, self.agent_y, self.agent_x))

        # update internal state
        self.agent_z = page
        self.agent_y = row
        self.agent_x = column

        reward = self.__get_reward(page, row, column) - (collision * self.collision_penalty)
        # if a == self.HOLD_POS:
        #     reward -= 5
        rewards = [reward]

        observation = self.__get_obs(page, row, column)
        observations = [observation]

        done = self.__is_terminal()

        return rewards, observations, done

    def render(self):
        """
        Output a human readable text representation of the environment.
        """
        for page in range(self.z_size):
            print("z = " + str(page))
            for row in range(self.y_size):
                line = ''
                for column in range(self.x_size):
                    # line += '['
                    symbol = '   '
                    # test for obstacle
                    if self.grid[page][row][column]:
                        # render obstacle in this square
                        symbol = '\u2588\u2588\u2588'

                    # test for goal
                    if self.goal_z == page and self.goal_y == row and self.goal_x == column:
                        # render goal in this square
                        symbol = ' G '

                    # test for agent
                    if self.agent_z == page and self.agent_y == row and self.agent_x == column:
                        # render agent in this square
                        symbol = ' A '

                    line += symbol
                    # line += ']'
                print(line)
            print('-' * self.x_size * 3)
        print('=' * self.x_size * 3)

    def render_to_file(self, file):
        """
        Write a human readable text representation of the environment to file.
        """
        for page in range(self.z_size):
            file.write("z = " + str(page) + "\n")
            for row in range(self.y_size):
                line = ''
                for column in range(self.x_size):
                    # line += '['
                    symbol = '   '
                    # test for obstacle
                    if self.grid[page][row][column]:
                        # render obstacle in this square
                        symbol = 'XXX'

                    # test for goal
                    if self.goal_z == page and self.goal_y == row and self.goal_x == column:
                        # render goal in this square
                        symbol = ' G '

                    # test for agent
                    if self.agent_z == page and self.agent_y == row and self.agent_x == column:
                        # render agent in this square
                        symbol = ' A '

                    line += symbol
                    # line += ']'
                file.write(line + "\n")
            file.write(('-' * self.x_size * 3) + "\n")
        file.write(('=' * self.x_size * 3) + "\n")

    def get_true_state(self):
        """
        Output a belief distribution with probability of 1 for the true state.
        :return: true state belief distribution
        """
        belief = np.zeros([self.z_size, self.y_size, self.x_size])
        belief[self.agent_z][self.agent_y][self.agent_x] = 1
        return belief

    def get_pomdp(self):
        """
        Return a POMDP description of the environment.
        :return: (T, R, Z)
        """
        return self.get_transition_model(), self.get_reward_model(), self.get_obs_model()

    def get_transition_model(self):
        """
        Compute underlying MDP transition model, T(s,a,s')

        Format = [n, m, l, a, n', m', l']
        :return: transition model
        """
        transition_model = np.zeros([self.z_size, self.y_size, self.x_size, self.NUM_ACTIONS,
                                     self.z_size, self.y_size, self.x_size])
        # loop across all non border cells
        # border cells can never be occupied, so don't care about these values
        for p in range(1, self.z_size - 1):
            for r in range(1, self.y_size - 1):
                for c in range(1, self.x_size - 1):
                    # if cell is obstacle, don't need to compute
                    if self.grid[p][r][c] == self.OBSTACLE:
                        continue

                    surplus_hold = 0
                    surplus_north = 0
                    surplus_south = 0
                    surplus_east = 0
                    surplus_west = 0
                    surplus_up = 0
                    surplus_down = 0

                    # transition probabilities for cell to north
                    if self.grid[p][r - 1][c] == self.FREE_SPACE:
                        # cell to north is free
                        transition_model[p][r][c][self.HOLD_POS][p][r - 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.NORTH][p][r - 1][c] = self.t_prob
                        transition_model[p][r][c][self.SOUTH][p][r - 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.EAST][p][r - 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.WEST][p][r - 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.ASCEND][p][r - 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.DESCEND][p][r - 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    else:
                        # cell to north is obstacle - add surplus probability for all actions
                        transition_model[p][r][c][self.HOLD_POS][p][r - 1][c] = 0
                        transition_model[p][r][c][self.NORTH][p][r - 1][c] = 0
                        transition_model[p][r][c][self.SOUTH][p][r - 1][c] = 0
                        transition_model[p][r][c][self.EAST][p][r - 1][c] = 0
                        transition_model[p][r][c][self.WEST][p][r - 1][c] = 0
                        surplus_hold += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_north += self.t_prob
                        surplus_south += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_east += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_west += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_up += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_down += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)

                    # transition probabilities for cell to south
                    if self.grid[p][r + 1][c] == self.FREE_SPACE:
                        # cell to south is free
                        transition_model[p][r][c][self.HOLD_POS][p][r + 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.NORTH][p][r + 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.SOUTH][p][r + 1][c] = self.t_prob
                        transition_model[p][r][c][self.EAST][p][r + 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.WEST][p][r + 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.ASCEND][p][r - 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.DESCEND][p][r - 1][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    else:
                        # cell to south is obstacle - add surplus probability for all actions
                        transition_model[p][r][c][self.HOLD_POS][p][r + 1][c] = 0
                        transition_model[p][r][c][self.NORTH][p][r + 1][c] = 0
                        transition_model[p][r][c][self.SOUTH][p][r + 1][c] = 0
                        transition_model[p][r][c][self.EAST][p][r + 1][c] = 0
                        transition_model[p][r][c][self.WEST][p][r + 1][c] = 0
                        transition_model[p][r][c][self.ASCEND][p][r + 1][c] = 0
                        transition_model[p][r][c][self.DESCEND][p][r + 1][c] = 0
                        surplus_hold += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_north += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_south += self.t_prob
                        surplus_east += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_west += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_up += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_down += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)

                    # transition probabilities for cell to east
                    if self.grid[p][r][c + 1] == self.FREE_SPACE:
                        # cell to east is free
                        transition_model[p][r][c][self.HOLD_POS][p][r][c + 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.NORTH][p][r][c + 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.SOUTH][p][r][c + 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.EAST][p][r][c + 1] = self.t_prob
                        transition_model[p][r][c][self.WEST][p][r][c + 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.ASCEND][p][r][c + 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.DESCEND][p][r][c + 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    else:
                        # cell to east is obstacle - add surplus probability for all actions
                        transition_model[p][r][c][self.HOLD_POS][p][r][c + 1] = 0
                        transition_model[p][r][c][self.NORTH][p][r][c + 1] = 0
                        transition_model[p][r][c][self.SOUTH][p][r][c + 1] = 0
                        transition_model[p][r][c][self.EAST][p][r][c + 1] = 0
                        transition_model[p][r][c][self.WEST][p][r][c + 1] = 0
                        transition_model[p][r][c][self.ASCEND][p][r][c + 1] = 0
                        transition_model[p][r][c][self.DESCEND][p][r][c + 1] = 0
                        surplus_hold += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_north += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_south += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_east += self.t_prob
                        surplus_west += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_up += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_down += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)

                    # transition probabilities for cell to west
                    if self.grid[p][r][c - 1] == self.FREE_SPACE:
                        # cell to west is free
                        transition_model[p][r][c][self.HOLD_POS][p][r][c - 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.NORTH][p][r][c - 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.SOUTH][p][r][c - 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.EAST][p][r][c - 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.WEST][p][r][c - 1] = self.t_prob
                        transition_model[p][r][c][self.ASCEND][p][r][c - 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.DESCEND][p][r][c - 1] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    else:
                        # cell to west is obstacle - add surplus probability for all actions
                        transition_model[p][r][c][self.HOLD_POS][p][r][c - 1] = 0
                        transition_model[p][r][c][self.NORTH][p][r][c - 1] = 0
                        transition_model[p][r][c][self.SOUTH][p][r][c - 1] = 0
                        transition_model[p][r][c][self.EAST][p][r][c - 1] = 0
                        transition_model[p][r][c][self.WEST][p][r][c - 1] = 0
                        transition_model[p][r][c][self.ASCEND][p][r][c - 1] = 0
                        transition_model[p][r][c][self.DESCEND][p][r][c - 1] = 0
                        surplus_hold += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_north += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_south += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_east += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_west += self.t_prob
                        surplus_up += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_down += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)

                    # transition probabilities for cell above
                    if self.grid[p - 1][r][c] == self.FREE_SPACE:
                        # cell above is free
                        transition_model[p][r][c][self.HOLD_POS][p - 1][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.NORTH][p - 1][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.SOUTH][p - 1][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.EAST][p - 1][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.WEST][p - 1][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.ASCEND][p - 1][r][c] = self.t_prob
                        transition_model[p][r][c][self.DESCEND][p - 1][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                    else:
                        # cell above is obstacle - add surplus probability for all actions
                        transition_model[p][r][c][self.HOLD_POS][p - 1][r][c] = 0
                        transition_model[p][r][c][self.NORTH][p - 1][r][c] = 0
                        transition_model[p][r][c][self.SOUTH][p - 1][r][c] = 0
                        transition_model[p][r][c][self.EAST][p - 1][r][c] = 0
                        transition_model[p][r][c][self.WEST][p - 1][r][c] = 0
                        transition_model[p][r][c][self.ASCEND][p - 1][r][c] = 0
                        transition_model[p][r][c][self.DESCEND][p - 1][r][c] = 0
                        surplus_hold += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_north += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_south += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_east += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_west += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_up += self.t_prob
                        surplus_down += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)

                    # transition probabilities for cell below
                    if self.grid[p + 1][r][c] == self.FREE_SPACE:
                        # cell below is free
                        transition_model[p][r][c][self.HOLD_POS][p + 1][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.NORTH][p + 1][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.SOUTH][p + 1][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.EAST][p + 1][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.WEST][p + 1][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.ASCEND][p + 1][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        transition_model[p][r][c][self.DESCEND][p + 1][r][c] = self.t_prob
                    else:
                        # cell below is obstacle - add surplus probability for all actions
                        transition_model[p][r][c][self.HOLD_POS][p - 1][r][c] = 0
                        transition_model[p][r][c][self.NORTH][p - 1][r][c] = 0
                        transition_model[p][r][c][self.SOUTH][p - 1][r][c] = 0
                        transition_model[p][r][c][self.EAST][p - 1][r][c] = 0
                        transition_model[p][r][c][self.WEST][p - 1][r][c] = 0
                        transition_model[p][r][c][self.ASCEND][p - 1][r][c] = 0
                        transition_model[p][r][c][self.DESCEND][p - 1][r][c] = 0
                        surplus_hold += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_north += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_south += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_east += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_west += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_up += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_down += self.t_prob

                    # transition probabilities for staying in the same cell
                    transition_model[p][r][c][self.HOLD_POS][p][r][c] = self.t_prob + surplus_hold
                    transition_model[p][r][c][self.NORTH][p][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1) + surplus_north
                    transition_model[p][r][c][self.SOUTH][p][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1) + surplus_south
                    transition_model[p][r][c][self.EAST][p][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1) + surplus_east
                    transition_model[p][r][c][self.WEST][p][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1) + surplus_west
                    transition_model[p][r][c][self.ASCEND][p][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1) + surplus_up
                    transition_model[p][r][c][self.DESCEND][p][r][c] = (1 - self.t_prob) / (self.NUM_ACTIONS - 1) + surplus_down

        return transition_model

    def get_reward_model(self):
        """
        Compute the underlying MDP reward model, R(s,a)

        Format = [n, m, a]
        :return: reward model
        """
        reward_model = np.ones([self.z_size, self.y_size, self.x_size, self.NUM_ACTIONS]) * -1

        # assign reward of +1 for hold position action in goal cell
        reward_model[self.goal_z][self.goal_y][self.goal_x][self.HOLD_POS] = 1

        # assign collision penalties for all colliding actions
        for p in range(1, self.z_size - 1):
            for r in range(1, self.y_size - 1):
                for c in range(1, self.x_size - 1):
                    # up direction
                    if self.grid[p - 1][r][c] == self.OBSTACLE:
                        reward_model[p][r][c][self.ASCEND] -= self.collision_penalty
                    # down direction
                    if self.grid[p + 1][r][c] == self.OBSTACLE:
                        reward_model[p][r][c][self.DESCEND] -= self.collision_penalty
                    # north direction
                    if self.grid[p][r - 1][c] == self.OBSTACLE:
                        reward_model[p][r][c][self.NORTH] -= self.collision_penalty
                    # south direction
                    if self.grid[p][r + 1][c] == self.OBSTACLE:
                        reward_model[p][r][c][self.SOUTH] -= self.collision_penalty
                    # east direction
                    if self.grid[p][r][c + 1] == self.OBSTACLE:
                        reward_model[p][r][c][self.EAST] -= self.collision_penalty
                    # west direction
                    if self.grid[p][r][c - 1] == self.OBSTACLE:
                        reward_model[p][r][c][self.WEST] -= self.collision_penalty

        # assign obstacles reward of zero
        for p in range(self.z_size):
            for r in range(self.y_size):
                for c in range(self.x_size):
                    if self.grid[p][r][c] == self.OBSTACLE:
                        for a in range(self.NUM_ACTIONS):
                            reward_model[p][r][c][a] = 0

        return reward_model

    def get_obs_model(self):
        """
        Compute the underlying MDP observation model, Z(o|s).

        Format = [n, m, l, o]
        :return: observation model
        """
        return None

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
        return self.get_true_state()

    def __path_len(self, z1, y1, x1, z2, y2, x2):
        container = [(z1, y1, x1, 0)]
        visited = np.zeros([self.z_size, self.y_size, self.x_size], dtype=np.int8)
        while np.size(container) > 0:
            z, y, x, d = container.pop(0)
            visited[z][y][x] = 1
            # check if at goal
            if (z == z2) and (y == y2) and (x == x2):
                return d    # indicates path length
            # check neighbour above
            if self.grid[z - 1][y][x] == self.FREE_SPACE and visited[z - 1][y][x] == 0:
                container.append((z - 1, y, x, d + 1))
            # check neighbour below
            if self.grid[z + 1][y][x] == self.FREE_SPACE and visited[z + 1][y][x] == 0:
                container.append((z + 1, y, x, d + 1))
            # check neighbour to north
            if self.grid[z][y - 1][x] == self.FREE_SPACE and visited[z][y - 1][x] == 0:
                container.append((z, y - 1, x, d + 1))
            # check neighbour to south
            if self.grid[z][y + 1][x] == self.FREE_SPACE and visited[z][y + 1][x] == 0:
                container.append((z, y + 1, x, d + 1))
            # check neighbour to west
            if self.grid[z][y][x - 1] == self.FREE_SPACE and visited[z][y][x - 1] == 0:
                container.append((z, y, x - 1, d + 1))
            # check neighbour to east
            if self.grid[z][y][x + 1] == self.FREE_SPACE and visited[z][y][x + 1] == 0:
                container.append((z, y, x + 1, d + 1))
        # no nodes left in container
        return -1   # indicates no path exists

    def __get_obs(self, z, y, x):
        return np.array([z, y, x])

    def __get_reward(self, z, y, x):
        # dist = math.sqrt(((y - self.goal_y) ** 2) + ((x - self.goal_x) ** 2))
        # return -1 * dist

        if z == self.goal_z and y == self.goal_y and x == self.goal_x:
            return 1.0
            # return 10.0
        else:
            return 0.0
            # return -1.0

    def __is_terminal(self):
        if self.agent_z == self.goal_z and self.agent_y == self.goal_y and self.agent_x == self.goal_x:
            return True
        else:
            return False

    @staticmethod
    def __hold_pos(position):
        """
        Do not move.
        :param position: (PG,ROW,COL)
        :return: (new PG, new ROW, new COL, collision [T/F])
        """
        p, r, c = position
        return p, r, c, False  # assumed to not be in collision

    def __move_north(self, position):
        """
        Attempt to move the agent with the given position in the North direction.
        :param position: (PG,ROW,COL)
        :return: (new PG, new ROW, new COL, collision [T/F])
        """
        p, r, c = position

        # check for out of bounds
        if r == 0:
            # can't move without hitting boundary
            return p, r, c, True

        # check for obstacle
        if self.grid[p][r - 1][c] == self.OBSTACLE:
            # can't move without hitting obstacle
            return p, r, c, True

        # move is possible without collision
        return p, r - 1, c, False

    def __move_south(self, position):
        """
        Attempt to move the agent with the given position in the South direction.
        :param position: (PG,ROW,COL)
        :return: (new PG, new ROW, new COL, collision [T/F])
        """
        p, r, c = position

        # check for out of bounds
        if r == self.y_size - 1:
            # can't move without hitting boundary
            return p, r, c, True

        # check for obstacle
        if self.grid[p][r + 1][c] == self.OBSTACLE:
            # can't move without hitting obstacle
            return p, r, c, True

        # move is possible without collision
        return p, r + 1, c, False

    def __move_east(self, position):
        """
        Attempt to move the agent with the given position in the East direction.
        :param position: (PG,ROW,COL)
        :return: (new PG, new ROW, new COL, collision [T/F])
        """
        p, r, c = position

        # check for out of bounds
        if c == self.x_size - 1:
            # can't move without hitting boundary
            return p, r, c, True

        # check for obstacle
        if self.grid[p][r][c + 1] == self.OBSTACLE:
            # can't move without hitting obstacle
            return p, r, c, True

        # move is possible without collision
        return p, r, c + 1, False

    def __move_west(self, position):
        """
        Attempt to move the agent with the given position in the West direction.
        :param position: (PG,ROW,COL)
        :return: (new PG, new ROW, new COL, collision [T/F])
        """
        p, r, c = position

        # check for out of bounds
        if c == 0:
            # can't move without hitting boundary
            return p, r, c, True

        # check for obstacle
        if self.grid[p][r][c - 1] == self.OBSTACLE:
            # can't move without hitting obstacle
            return p, r, c, True

        # move is possible without collision
        return p, r, c - 1, False

    def __move_up(self, position):
        """
        Attempt to move the agent with the given position in the UP direction.
        :param position: (PG,ROW,COL)
        :return: (new PG, new ROW, new COL, collision [T/F])
        """
        p, r, c = position

        # check for out of bounds
        if p == 0:
            # can't move without hitting boundary
            return p, r, c, True

        # check for obstacle
        if self.grid[p-1][r][c] == self.OBSTACLE:
            # can't move without hitting obstacle
            return p, r, c, True

        # move is possible without collision
        return p - 1, r, c, False

    def __move_down(self, position):
        """
        Attempt to move the agent with the given position in the DOWN direction.
        :param position: (PG,ROW,COL)
        :return: (new PG, new ROW, new COL, collision [T/F])
        """
        p, r, c = position

        # check for out of bounds
        if p == self.z_size - 1:
            # can't move without hitting boundary
            return p, r, c, True

        # check for obstacle
        if self.grid[p+1][r][c] == self.OBSTACLE:
            # can't move without hitting obstacle
            return p, r, c, True

        # move is possible without collision
        return p + 1, r, c, False


def combination(n, r):
    return np.math.factorial(n) / (np.math.factorial(n - r) * np.math.factorial(r))






