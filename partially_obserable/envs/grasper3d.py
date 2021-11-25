import numpy as np
import random
import time
from itertools import compress


class Grasper3D:
    """
    Class representing a parameterised 3D grasping environment.

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

    MIN_DENSITY = 0.3
    MAX_DENSITY = 0.8

    def __init__(self, x_size, y_size, z_size, obj_max_x, obj_max_y, obj_max_z, obs_dirs, obstacle_prob,
                 transition_prob, obs_success_prob, collision_penalty):

        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.obj_max_x = obj_max_x
        self.obj_max_y = obj_max_y
        self.obj_max_z = obj_max_z
        self.obs_dirs = obs_dirs    # list of tuples, each giving a relative position
        self.obstacle_prob = obstacle_prob
        self.t_prob = transition_prob
        self.obs_success_prob = obs_success_prob
        self.collision_penalty = collision_penalty

        self.grid = None
        self.grid_padded = None
        self.rewards = None
        self.config_space = None

        self.object_grid = None
        self.object_reward = None
        self.object_depth = None
        self.object_length = None
        self.object_width = None

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
        return np.array([[2 for i in self.obs_dirs]])

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
        return [self.__get_obs(self.agent_z, self.agent_y, self.agent_x)]

    def reset(self):
        # generate a new "random" grid instance
        np.random.seed(int(time.time()))

        # %%%%% make graspable object %%%%%
        # randomise size within lower and upper limits
        obj_depth = random.randint(int(self.obj_max_z / 2), self.obj_max_z)
        obj_length = random.randint(int(self.obj_max_y / 2), self.obj_max_y)
        obj_width = random.randint(int(self.obj_max_x / 2), self.obj_max_x)

        # generate obstacle shape
        while True:
            density = (random.random() * (self.MAX_DENSITY - self.MIN_DENSITY)) + self.MIN_DENSITY
            object_grid = np.zeros([obj_depth, obj_length, obj_width], dtype=np.int8)
            rand_field = np.random.rand(obj_depth, obj_length, obj_width)
            object_grid = np.array(np.logical_or(object_grid, (rand_field < density)), 'i')
            object_reward = np.zeros([obj_depth, obj_length, obj_width], dtype=np.int8)

            # check that object is fully connected
            partitions = []
            for p in range(obj_depth):
                for r in range(obj_length):
                    for c in range(obj_width):
                        # skip if cell not occupied
                        if object_grid[p][r][c] == self.FREE_SPACE:
                            continue

                        # check for adjacency to existing partitions
                        in_partition = [False for _ in partitions]
                        for i in range(len(partitions)):
                            part = partitions[i]
                            if ((p-1, r, c) in part or (p+1, r, c) in part or
                                    (p, r-1, c) in part or (p, r+1, c) in part or
                                    (p, r, c-1) in part or (p, r, c+1) in part):
                                partitions[i].append((p, r, c))
                                in_partition[i] = True

                        # create new partition if not adjacent
                        if not any(in_partition):
                            partitions.append([(p, r, c)])
                            in_partition.append(True)

                        # merge partitions if (p,r,c) is in multiple
                        assert len(partitions) == len(in_partition)
                        to_merge = list(compress(partitions, in_partition))

                        if len(to_merge) > 0:
                            for part in to_merge:
                                partitions.remove(part)     # remove original partition
                                part.remove((p, r, c))      # remove duplicates of current cell
                            new_part = sum(to_merge, [])    # union of partitions
                            new_part.append((p, r, c))      # add back current cell
                            partitions.append(new_part)

            # not fully connected - attempt 1-bridge merges
            while len(partitions) > 1:
                exit_loop = False
                new_partitions = partitions[:]
                for i in range(len(partitions) - 1):
                    for j in range(i + 1, len(partitions)):
                        part1 = partitions[i]
                        part2 = partitions[j]
                        for (p, r, c) in part1:
                            # straight bridge merges
                            if (p-2, r, c) in part2:
                                # add bridge and merge partitions
                                object_grid[p-1][r][c] = self.OBSTACLE
                                bridge = (p-1, r, c)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            elif (p+2, r, c) in part2:
                                # add bridge and merge partitions
                                object_grid[p+1][r][c] = self.OBSTACLE
                                bridge = (p+1, r, c)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            elif (p, r-2, c) in part2:
                                # add bridge and merge partitions
                                object_grid[p][r-1][c] = self.OBSTACLE
                                bridge = (p, r-1, c)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            elif (p, r+2, c) in part2:
                                # add bridge and merge partitions
                                object_grid[p][r+1][c] = self.OBSTACLE
                                bridge = (p, r+1, c)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            elif (p, r, c-2) in part2:
                                # add bridge and merge partitions
                                object_grid[p][r][c-1] = self.OBSTACLE
                                bridge = (p, r, c-1)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            elif (p, r, c+2) in part2:
                                # add bridge and merge partitions
                                object_grid[p][r][c+1] = self.OBSTACLE
                                bridge = (p, r, c+1)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            # corner bridge merges
                            elif (p-1, r-1, c) in part2:
                                # add bridge and merge partitions
                                if random.random() < 0.5:
                                    object_grid[p-1][r][c] = self.OBSTACLE
                                    bridge = (p-1, r, c)
                                else:
                                    object_grid[p][r-1][c] = self.OBSTACLE
                                    bridge = (p, r-1, c)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            elif (p+1, r+1, c) in part2:
                                # add bridge and merge partitions
                                if random.random() < 0.5:
                                    object_grid[p+1][r][c] = self.OBSTACLE
                                    bridge = (p+1, r, c)
                                else:
                                    object_grid[p][r+1][c] = self.OBSTACLE
                                    bridge = (p, r+1, c)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            elif (p+1, r-1, c) in part2:
                                # add bridge and merge partitions
                                if random.random() < 0.5:
                                    object_grid[p+1][r][c] = self.OBSTACLE
                                    bridge = (p+1, r, c)
                                else:
                                    object_grid[p][r-1][c] = self.OBSTACLE
                                    bridge = (p, r-1, c)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            elif (p-1, r+1, c) in part2:
                                # add bridge and merge partitions
                                if random.random() < 0.5:
                                    object_grid[p-1][r][c] = self.OBSTACLE
                                    bridge = (p-1, r, c)
                                else:
                                    object_grid[p][r+1][c] = self.OBSTACLE
                                    bridge = (p, r+1, c)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            elif (p-1, r, c-1) in part2:
                                # add bridge and merge partitions
                                if random.random() < 0.5:
                                    object_grid[p-1][r][c] = self.OBSTACLE
                                    bridge = (p-1, r, c)
                                else:
                                    object_grid[p][r][c-1] = self.OBSTACLE
                                    bridge = (p, r, c-1)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            elif (p+1, r, c+1) in part2:
                                # add bridge and merge partitions
                                if random.random() < 0.5:
                                    object_grid[p+1][r][c] = self.OBSTACLE
                                    bridge = (p+1, r, c)
                                else:
                                    object_grid[p][r][c+1] = self.OBSTACLE
                                    bridge = (p, r, c+1)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            elif (p+1, r, c-1) in part2:
                                # add bridge and merge partitions
                                if random.random() < 0.5:
                                    object_grid[p+1][r][c] = self.OBSTACLE
                                    bridge = (p+1, r, c)
                                else:
                                    object_grid[p][r][c-1] = self.OBSTACLE
                                    bridge = (p, r, c-1)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            elif (p-1, r, c+1) in part2:
                                # add bridge and merge partitions
                                if random.random() < 0.5:
                                    object_grid[p-1][r][c] = self.OBSTACLE
                                    bridge = (p-1, r, c)
                                else:
                                    object_grid[p][r][c+1] = self.OBSTACLE
                                    bridge = (p, r, c+1)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            elif (p, r-1, c-1) in part2:
                                # add bridge and merge partitions
                                if random.random() < 0.5:
                                    object_grid[p][r-1][c] = self.OBSTACLE
                                    bridge = (p, r-1, c)
                                else:
                                    object_grid[p][r][c-1] = self.OBSTACLE
                                    bridge = (p, r, c-1)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            elif (p, r+1, c+1) in part2:
                                # add bridge and merge partitions
                                if random.random() < 0.5:
                                    object_grid[p][r+1][c] = self.OBSTACLE
                                    bridge = (p, r+1, c)
                                else:
                                    object_grid[p][r][c+1] = self.OBSTACLE
                                    bridge = (p, r, c+1)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            elif (p, r+1, c-1) in part2:
                                # add bridge and merge partitions
                                if random.random() < 0.5:
                                    object_grid[p][r+1][c] = self.OBSTACLE
                                    bridge = (p, r+1, c)
                                else:
                                    object_grid[p][r][c-1] = self.OBSTACLE
                                    bridge = (p, r, c-1)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break
                            elif (p, r-1, c+1) in part2:
                                # add bridge and merge partitions
                                if random.random() < 0.5:
                                    object_grid[p][r-1][c] = self.OBSTACLE
                                    bridge = (p, r-1, c)
                                else:
                                    object_grid[p][r][c+1] = self.OBSTACLE
                                    bridge = (p, r, c+1)
                                new_partitions.remove(part1)
                                new_partitions.remove(part2)
                                new_partitions.append(part1 + part2 + [bridge])
                                exit_loop = True
                                break

                        if exit_loop:
                            break
                    if exit_loop:
                        break
                if exit_loop:
                    partitions = new_partitions
                    continue

                # all possible merges have been made and still disconnected - remove all but largest connected cluster
                lengths = [len(part) for part in partitions]
                max_idx = lengths.index(max(lengths))
                for i in range(len(partitions)):
                    if i != max_idx:
                        for (p, r, c) in partitions[i]:
                            object_grid[p][r][c] = self.FREE_SPACE
                break

            # remove cavities
            for p in range(1, obj_depth - 1):
                for r in range(1, obj_length - 1):
                    for c in range(1, obj_width - 1):
                        if (object_grid[p][r][c] == self.FREE_SPACE and
                                object_grid[p-1][r][c] == self.OBSTACLE and object_grid[p+1][r][c] == self.OBSTACLE and
                                object_grid[p][r-1][c] == self.OBSTACLE and object_grid[p][r+1][c] == self.OBSTACLE and
                                object_grid[p][r][c-1] == self.OBSTACLE and object_grid[p][r][c+1] == self.OBSTACLE):
                            # surrounded on all sides -> fill cell
                            object_grid[p][r][c] = self.OBSTACLE

            # find and mark feasible grasp points
            num_points = 0
            object_grid_padded = np.pad(object_grid, [(1, 1), (1, 1), (1, 1)], mode='constant')
            for p in range(obj_depth):
                for r in range(obj_length):
                    for c in range(obj_width):
                        p1 = p + 1
                        r1 = r + 1
                        c1 = c + 1
                        if (object_grid_padded[p1][r1][c1] == self.OBSTACLE and
                                object_grid_padded[p1][r1][c1-1] == self.FREE_SPACE and
                                object_grid_padded[p1][r1][c1+1] == self.FREE_SPACE and
                                object_grid_padded[p1-1][r1][c1] == self.FREE_SPACE and
                                object_grid_padded[p1-1][r1][c1-1] == self.FREE_SPACE and
                                object_grid_padded[p1-1][r1][c1+1] == self.FREE_SPACE):
                            # feasible grasp point found
                            object_reward[p][r][c] = 1
                            num_points += 1

            # start again if there are no feasible grasp points
            if num_points == 0:
                continue

            # exit only once all conditions satisfied
            break

        # store object parameters
        self.object_grid = object_grid
        self.object_reward = object_reward
        self.object_depth = obj_depth
        self.object_length = obj_length
        self.object_width = obj_width

        # %%%%% make workspace (including reward image) %%%%%
        z_offset = self.z_size - obj_depth - 1  # place object on floor
        z_rem = 1
        y_offset = random.randint(2, self.y_size - obj_length - 2)
        y_rem = self.y_size - y_offset - obj_length
        x_offset = random.randint(2, self.x_size - obj_width - 2)
        x_rem = self.x_size - x_offset - obj_width

        # place object in workspace at random position
        self.grid = np.pad(object_grid, [(z_offset, z_rem), (y_offset, y_rem), (x_offset, x_rem)], mode='constant')
        self.rewards = np.pad(object_reward, [(z_offset, z_rem), (y_offset, y_rem), (x_offset, x_rem)], mode='constant')

        # add border
        self.grid[0, :, :] = self.OBSTACLE
        self.grid[-1, :, :] = self.OBSTACLE
        self.grid[:, 0, :] = self.OBSTACLE
        self.grid[:, -1, :] = self.OBSTACLE
        self.grid[:, :, 0] = self.OBSTACLE
        self.grid[:, :, -1] = self.OBSTACLE

        # %%%%% generate initial position %%%%%
        self.agent_z = random.randint(2, self.z_size - obj_depth - 1)
        self.agent_y = random.randint(2, self.y_size - 2)
        self.agent_x = random.randint(2, self.x_size - 2)
        self.initial_z = self.agent_z
        self.initial_y = self.agent_y
        self.initial_x = self.agent_x

        # %%%%% get forbidden regions in c-space %%%%%
        # pad to get border with thickness of 2
        grid_padded = np.pad(self.grid, [(1, 1), (1, 1), (1, 1)], mode='constant')
        grid_padded[0, :, :] = self.OBSTACLE
        grid_padded[-1, :, :] = self.OBSTACLE
        grid_padded[:, 0, :] = self.OBSTACLE
        grid_padded[:, -1, :] = self.OBSTACLE
        grid_padded[:, :, 0] = self.OBSTACLE
        grid_padded[:, :, -1] = self.OBSTACLE
        # store grid_padded for observation generation
        self.grid_padded = grid_padded

        # sum collision areas together
        grid_temp = sum([np.roll(grid_padded, shift=1, axis=0),
                         np.roll(grid_padded, shift=-1, axis=2),
                         np.roll(grid_padded, shift=1, axis=2),
                         np.roll(np.roll(grid_padded, shift=1, axis=0), shift=-1, axis=2),
                         np.roll(np.roll(grid_padded, shift=1, axis=0), shift=1, axis=2)])
        # remove extra border
        grid_temp = grid_temp[1:-1, 1:-1, 1:-1]
        # clip to reduce to free space and obstacle in config space
        grid_temp = np.clip(grid_temp, 0, 1)

        self.config_space = grid_temp

    def regen_start_and_goal_pos(self):
        """
        Re-generate object position within workspace and initial grasper position
        """
        object_grid = self.object_grid
        object_reward = self.object_reward
        obj_depth = self.object_depth
        obj_length = self.object_length
        obj_width = self.object_width

        # %%%%% make workspace (including reward image) %%%%%
        z_offset = self.z_size - obj_depth - 1  # place object on floor
        z_rem = 1
        y_offset = random.randint(2, self.y_size - obj_length - 2)
        y_rem = self.y_size - y_offset - obj_length
        x_offset = random.randint(2, self.x_size - obj_width - 2)
        x_rem = self.x_size - x_offset - obj_width

        # place object in workspace at random position
        self.grid = np.pad(object_grid, [(z_offset, z_rem), (y_offset, y_rem), (x_offset, x_rem)], mode='constant')
        self.rewards = np.pad(object_reward, [(z_offset, z_rem), (y_offset, y_rem), (x_offset, x_rem)], mode='constant')

        # add border
        self.grid[0, :, :] = self.OBSTACLE
        self.grid[-1, :, :] = self.OBSTACLE
        self.grid[:, 0, :] = self.OBSTACLE
        self.grid[:, -1, :] = self.OBSTACLE
        self.grid[:, :, 0] = self.OBSTACLE
        self.grid[:, :, -1] = self.OBSTACLE

        # %%%%% generate initial position %%%%%
        self.agent_z = random.randint(2, self.z_size - obj_depth - 1)
        self.agent_y = random.randint(2, self.y_size - 2)
        self.agent_x = random.randint(2, self.x_size - 2)
        self.initial_z = self.agent_z
        self.initial_y = self.agent_y
        self.initial_x = self.agent_x

        # %%%%% get forbidden regions in c-space %%%%%
        # pad to get border with thickness of 2
        grid_padded = np.pad(self.grid, [(1, 1), (1, 1), (1, 1)], mode='constant')
        grid_padded[0, :, :] = self.OBSTACLE
        grid_padded[-1, :, :] = self.OBSTACLE
        grid_padded[:, 0, :] = self.OBSTACLE
        grid_padded[:, -1, :] = self.OBSTACLE
        grid_padded[:, :, 0] = self.OBSTACLE
        grid_padded[:, :, -1] = self.OBSTACLE
        # store grid_padded for observation generation
        self.grid_padded = grid_padded

        # sum collision areas together
        grid_temp = sum([np.roll(grid_padded, shift=1, axis=0),
                         np.roll(grid_padded, shift=-1, axis=2),
                         np.roll(grid_padded, shift=1, axis=2),
                         np.roll(np.roll(grid_padded, shift=1, axis=0), shift=-1, axis=2),
                         np.roll(np.roll(grid_padded, shift=1, axis=0), shift=1, axis=2)])
        # remove extra border
        grid_temp = grid_temp[1:-1, 1:-1, 1:-1]
        # clip to reduce to free space and obstacle in config space
        grid_temp = np.clip(grid_temp, 0, 1)

        self.config_space = grid_temp

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
                    if self.rewards[page][row][column] == 1:
                        # render feasible grasp point in this square
                        symbol = ' G '

                    # test for agent
                    if ((self.agent_z == page and self.agent_y == row and self.agent_x == column) or
                            (self.agent_z == page and self.agent_y == row and self.agent_x == column - 1) or
                            (self.agent_z == page and self.agent_y == row and self.agent_x == column + 1) or
                            (self.agent_z == page - 1 and self.agent_y == row and self.agent_x == column - 1) or
                            (self.agent_z == page - 1 and self.agent_y == row and self.agent_x == column) or
                            (self.agent_z == page - 1 and self.agent_y == row and self.agent_x == column + 1)):
                        # render grasper in this square
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
                        if self.rewards[page][row][column] == 1:
                            # render feasible grasp point in this square
                            symbol = ' G '

                        # test for agent
                        if ((self.agent_z == page and self.agent_y == row and self.agent_x == column) or
                                (self.agent_z == page and self.agent_y == row and self.agent_x == column - 1) or
                                (self.agent_z == page and self.agent_y == row and self.agent_x == column + 1) or
                                (self.agent_z == page - 1 and self.agent_y == row and self.agent_x == column - 1) or
                                (self.agent_z == page - 1 and self.agent_y == row and self.agent_x == column) or
                                (self.agent_z == page - 1 and self.agent_y == row and self.agent_x == column + 1)):
                            # render grasper in this square
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
                    if self.config_space[p][r][c] == self.OBSTACLE:
                        continue

                    surplus_hold = 0
                    surplus_north = 0
                    surplus_south = 0
                    surplus_east = 0
                    surplus_west = 0
                    surplus_up = 0
                    surplus_down = 0

                    # transition probabilities for cell to north
                    if self.config_space[p][r - 1][c] == self.FREE_SPACE:
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
                        transition_model[p][r][c][self.ASCEND][p][r - 1][c] = 0
                        transition_model[p][r][c][self.DESCEND][p][r - 1][c] = 0
                        surplus_hold += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_north += self.t_prob
                        surplus_south += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_east += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_west += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_up += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)
                        surplus_down += (1 - self.t_prob) / (self.NUM_ACTIONS - 1)

                    # transition probabilities for cell to south
                    if self.config_space[p][r + 1][c] == self.FREE_SPACE:
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
                    if self.config_space[p][r][c + 1] == self.FREE_SPACE:
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
                    if self.config_space[p][r][c - 1] == self.FREE_SPACE:
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
                    if self.config_space[p - 1][r][c] == self.FREE_SPACE:
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
                    if self.config_space[p + 1][r][c] == self.FREE_SPACE:
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

        # assign reward of +1 for hold position action at all feasible grasp points
        for p in range(1, self.z_size - 1):
            for r in range(1, self.y_size - 1):
                for c in range(1, self.x_size - 1):
                    if self.rewards[p][r][c] == 1:
                        reward_model[p][r][c][self.HOLD_POS] = 1

        # assign collision penalties for all colliding actions
        for p in range(1, self.z_size - 1):
            for r in range(1, self.y_size - 1):
                for c in range(1, self.x_size - 1):
                    # up direction
                    if self.config_space[p-1][r][c] == self.OBSTACLE:
                        reward_model[p][r][c][self.ASCEND] -= self.collision_penalty
                    # down direction
                    if self.config_space[p+1][r][c] == self.OBSTACLE:
                        reward_model[p][r][c][self.DESCEND] -= self.collision_penalty
                    # north direction
                    if self.config_space[p][r-1][c] == self.OBSTACLE:
                        reward_model[p][r][c][self.NORTH] -= self.collision_penalty
                    # south direction
                    if self.config_space[p][r+1][c] == self.OBSTACLE:
                        reward_model[p][r][c][self.SOUTH] -= self.collision_penalty
                    # east direction
                    if self.config_space[p][r][c+1] == self.OBSTACLE:
                        reward_model[p][r][c][self.EAST] -= self.collision_penalty
                    # west direction
                    if self.config_space[p][r][c-1] == self.OBSTACLE:
                        reward_model[p][r][c][self.WEST] -= self.collision_penalty

        # assign obstacles reward of zero
        for p in range(self.z_size):
            for r in range(self.y_size):
                for c in range(self.x_size):
                    if self.config_space[p][r][c] == self.OBSTACLE:
                        # this cell is in collision in c-space
                        for a in range(self.NUM_ACTIONS):
                            reward_model[p][r][c][a] = 0

        return reward_model

    def get_obs_model(self):
        """
        Compute the underlying MDP observation model, Z(o|s).

        Format = [n, m, l, o]
        :return: observation model
        """
        num_dirs = len(self.obs_dirs)
        num_obs = np.power(2, len(self.obs_dirs))
        # enumerate observations
        observations = []
        for o_1 in [self.FREE_SPACE, self.OBSTACLE]:
            for o_2 in [self.FREE_SPACE, self.OBSTACLE]:
                for o_3 in [self.FREE_SPACE, self.OBSTACLE]:
                    for o_4 in [self.FREE_SPACE, self.OBSTACLE]:
                        for o_5 in [self.FREE_SPACE, self.OBSTACLE]:
                            for o_6 in [self.FREE_SPACE, self.OBSTACLE]:
                                for o_7 in [self.FREE_SPACE, self.OBSTACLE]:
                                    observations.append([o_7, o_6, o_5, o_4, o_3, o_2, o_1])

        # enumerate probabilities of receiving 0, 1, ... num_dirs correct observations
        p_correct = [combination(num_dirs, i) *
                     np.power(self.obs_success_prob, i) *
                     np.power(1 - self.obs_success_prob, num_dirs - i) for i in range(num_dirs + 1)]

        obs_model = np.zeros([self.z_size, self.y_size, self.x_size, num_obs])

        # loop over all states (except borders)
        for p in range(2, self.z_size - 1):
            for r in range(1, self.y_size - 1):
                for c in range(2, self.x_size - 2):
                    # if cell is obstacle, Z(o|s) = 0 for all o
                    if self.grid[p][r][c] == self.OBSTACLE:
                        for o in range(num_obs):
                            obs_model[p][r][c][o] = 0.0
                    else:
                        # loop over all observations
                        for o in range(num_obs):
                            # find the number of directions for which the state is consistent with the observation
                            num_correct = 0
                            for i in range(len(self.obs_dirs)):
                                direction = self.obs_dirs[i]
                                p1 = p + direction[0]
                                r1 = r + direction[1]
                                c1 = c + direction[2]
                                if self.grid[p1][r1][c1] == observations[o][i]:
                                    # state is consistent with observation in this direction
                                    num_correct += 1
                            obs_model[p][r][c][o] = p_correct[num_correct]

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
        # generate a random distribution across the set of free cells
        grid_temp = np.array(self.config_space)
        grid_temp[self.initial_z][self.initial_y][self.initial_x] = self.OBSTACLE   # don't include initial state
        free_states = np.nonzero((grid_temp == self.FREE_SPACE).flatten())[0]
        freespace_size = len(free_states)

        b0sizes = np.floor(freespace_size / np.power(2.0, np.arange(20)))
        b0sizes = b0sizes[:np.nonzero(b0sizes < 1)[0][0]]
        b0size = int(np.random.choice(b0sizes))

        b0ind = np.random.choice(free_states, b0size, replace=False)
        b0 = np.zeros([self.z_size * self.y_size * self.x_size])
        b0[b0ind] = 1.0 / (b0size + 1)  # uniform distribution over sampled states

        b0 = np.reshape(b0, [self.z_size, self.y_size, self.x_size])
        b0[self.initial_z][self.initial_y][self.initial_x] = 1.0 / (b0size + 1)     # add init state to init belief set
        return b0

    def __get_obs(self, z, y, x):
        obs = []
        for dz, dy, dx in self.obs_dirs:
            if random.random() < self.obs_success_prob:
                # correct observation
                o = self.grid_padded[z + dz + 1][y + dy + 1][x + dx + 1]
            else:
                # incorrect observation
                o = 1 - self.grid_padded[z + dz + 1][y + dy + 1][x + dx + 1]
            obs.append(o)
        return np.array(obs)

    def __get_reward(self, z, y, x):
        # dist = math.sqrt(((y - self.goal_y) ** 2) + ((x - self.goal_x) ** 2))
        # return -1 * dist
        if self.rewards[z][y][x] == 1:
            return 1.0
            # return 10.0
        else:
            return 0.0
            # return -1.0

    def __is_terminal(self):
        if self.rewards[self.agent_z][self.agent_y][self.agent_x] == 1:
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

        # check for obstacle
        if self.config_space[p][r - 1][c] == self.OBSTACLE:
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

        # check for obstacle
        if self.config_space[p][r + 1][c] == self.OBSTACLE:
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

        # check for obstacle
        if self.config_space[p][r][c + 1] == self.OBSTACLE:
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

        # check for obstacle
        if self.config_space[p][r][c - 1] == self.OBSTACLE:
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

        # check for obstacle
        if self.config_space[p - 1][r][c] == self.OBSTACLE:
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

        # check for obstacle
        if self.config_space[p + 1][r][c] == self.OBSTACLE:
            # can't move without hitting obstacle
            return p, r, c, True

        # move is possible without collision
        return p + 1, r, c, False


def combination(n, r):
    return np.math.factorial(n) / (np.math.factorial(n - r) * np.math.factorial(r))






