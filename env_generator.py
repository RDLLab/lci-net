import sys
import random
import numpy as np

# fully observable
from fully_observable.envs.full_obs_grid import FullObsGrid
from fully_observable.envs.full_obs_grid3d import FullObsGrid3D
from fully_observable.envs.full_obs_dynmaze_v1 import FullObsDynMazeV1
from fully_observable.envs.full_obs_dynmaze_v2 import FullObsDynMazeV2
from fully_observable.vi_expert import VIExpert

# partially observable
from partially_obserable.envs.grid2d import Grid2D
from partially_obserable.envs.grid3d import Grid3D
from partially_obserable.envs.dynmaze_v1 import DynMazeV1
from partially_obserable.envs.dynmaze_v2 import DynMazeV2
from partially_obserable.envs.grasper2d import Grasper2D
from partially_obserable.envs.grasper3d import Grasper3D
from partially_obserable.qmdp_expert import QMDPExpert


def main(arglist):
    env_type = arglist[0]
    num_training_envs = int(arglist[1])
    num_trajs_per_env = int(arglist[2])
    suffix = arglist[3]

    if 'full_obs' in env_type:
        expert_class = VIExpert
    else:
        expert_class = QMDPExpert

    np.set_printoptions(threshold=sys.maxsize)

    if env_type == "grid_det":
        size = 10
        s_dims = [size, size]
        moves = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]
        num_action = len(moves)
        stay_action = 4
        k_dims = [3, 3]
        observe_directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        obs_len = len(observe_directions)
        num_obs = 17
        num_outcomes = 17

        env = Grid2D(x_size=size,
                     y_size=size,
                     obs_dirs=observe_directions,
                     obstacle_prob=0.25,
                     transition_prob=1.0,
                     obs_success_prob=1.0,
                     collision_penalty=1)

    elif env_type == "grid_nondet":
        size = 10
        s_dims = [size, size]
        moves = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]
        num_action = len(moves)
        stay_action = 4
        k_dims = [3, 3]
        observe_directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        obs_len = len(observe_directions)
        num_obs = 17
        num_outcomes = 17

        env = Grid2D(x_size=size,
                     y_size=size,
                     obs_dirs=observe_directions,
                     obstacle_prob=0.25,
                     transition_prob=0.8,
                     obs_success_prob=0.9,
                     collision_penalty=1)

    elif env_type == 'grid_3d_det':
        size = 7
        s_dims = [size, size, size]
        moves = [[0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0], [-1, 0, 0], [1, 0, 0], [0, 0, 0]]
        num_action = len(moves)
        stay_action = 6
        k_dims = [3, 3, 3]
        observe_directions = [[0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
        obs_len = len(observe_directions)
        num_obs = 64
        num_outcomes = 64

        env = Grid3D(x_size=size,
                     y_size=size,
                     z_size=size,
                     obs_dirs=observe_directions,
                     obstacle_prob=0.25,
                     transition_prob=1.0,
                     obs_success_prob=1.0,
                     collision_penalty=10)

    elif env_type == 'grid_3d_nondet':
        size = 7
        s_dims = [size, size, size]
        moves = [[0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0], [-1, 0, 0], [1, 0, 0], [0, 0, 0]]
        num_action = len(moves)
        stay_action = 6
        k_dims = [3, 3, 3]
        observe_directions = [[0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
        obs_len = len(observe_directions)
        num_obs = 64
        num_outcomes = 64

        env = Grid3D(x_size=size,
                     y_size=size,
                     z_size=size,
                     obs_dirs=observe_directions,
                     obstacle_prob=0.25,
                     transition_prob=0.8,
                     obs_success_prob=0.9,
                     collision_penalty=10)

    elif env_type == "grid_det_large":
        size = 20
        s_dims = [size, size]
        moves = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]
        num_action = len(moves)
        stay_action = 4
        k_dims = [3, 3]
        observe_directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        obs_len = len(observe_directions)
        num_obs = 17
        num_outcomes = 16

        env = Grid2D(x_size=size,
                     y_size=size,
                     obs_dirs=observe_directions,
                     obstacle_prob=0.25,
                     transition_prob=1.0,
                     obs_success_prob=1.0,
                     collision_penalty=10)

    elif env_type == "grid_nondet_large":
        size = 20
        s_dims = [size, size]
        moves = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]
        num_action = len(moves)
        stay_action = 4
        k_dims = [3, 3]
        observe_directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        obs_len = len(observe_directions)
        num_obs = 17
        num_outcomes = 16

        env = Grid2D(x_size=size,
                     y_size=size,
                     obs_dirs=observe_directions,
                     obstacle_prob=0.25,
                     transition_prob=0.8,
                     obs_success_prob=0.9,
                     collision_penalty=1)

    elif env_type == "grasper2d_det":
        size = 14
        obj_size = 6
        s_dims = [size, size]
        moves = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]
        num_action = len(moves)
        stay_action = 4
        k_dims = [3, 3]
        observe_directions = [[0, -2], [0, 0], [0, 0], [0, 2], [1, -1], [1, 1]]
        obs_len = len(observe_directions)
        num_obs = 64
        num_outcomes = 64

        env = Grasper2D(x_size=size,
                        y_size=size,
                        obj_max_x=obj_size,
                        obj_max_y=obj_size,
                        obs_dirs=observe_directions,
                        obstacle_prob=0.25,
                        transition_prob=1.0,
                        obs_success_prob=1.0,
                        collision_penalty=1)

    elif env_type == "grasper2d_nondet":
        size = 14
        obj_size = 6
        s_dims = [size, size]
        moves = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]
        num_action = len(moves)
        stay_action = 4
        k_dims = [3, 3]
        observe_directions = [[0, -2], [0, 0], [0, 0], [0, 2], [1, -1], [1, 1]]
        obs_len = len(observe_directions)
        num_obs = 64
        num_outcomes = 64

        env = Grasper2D(x_size=size,
                        y_size=size,
                        obj_max_x=obj_size,
                        obj_max_y=obj_size,
                        obs_dirs=observe_directions,
                        obstacle_prob=0.25,
                        transition_prob=0.8,
                        obs_success_prob=0.9,
                        collision_penalty=1)

    elif env_type == "grasper3d_det":
        size = 7
        obj_size = 3
        s_dims = [size, size, size]
        moves = [[0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0], [-1, 0, 0], [1, 0, 0], [0, 0, 0]]
        num_action = len(moves)
        stay_action = 6
        k_dims = [3, 3, 3]
        observe_directions = [[0, 0, -2], [0, -1, -1], [0, 1, -1], [0, 0, 0],
                              [0, 0, 2], [0, -1, 1], [0, 1, 1]]
        obs_len = len(observe_directions)
        num_obs = 128
        num_outcomes = 128

        env = Grasper3D(x_size=size,
                        y_size=size,
                        z_size=size,
                        obj_max_x=obj_size,
                        obj_max_y=obj_size,
                        obj_max_z=obj_size,
                        obs_dirs=observe_directions,
                        obstacle_prob=0.25,
                        transition_prob=1.0,
                        obs_success_prob=1.0,
                        collision_penalty=1)

    elif env_type == "grasper3d_nondet":
        size = 7
        obj_size = 3
        s_dims = [size, size, size]
        moves = [[0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0], [-1, 0, 0], [1, 0, 0], [0, 0, 0]]
        num_action = len(moves)
        stay_action = 6
        k_dims = [3, 3, 3]
        observe_directions = [[0, 0, -2], [0, -1, -1], [0, 1, -1], [0, 0, 0],
                              [0, 0, 2], [0, -1, 1], [0, 1, 1]]
        obs_len = len(observe_directions)
        num_obs = 128
        num_outcomes = 128

        env = Grasper3D(x_size=size,
                        y_size=size,
                        z_size=size,
                        obj_max_x=obj_size,
                        obj_max_y=obj_size,
                        obj_max_z=obj_size,
                        obs_dirs=observe_directions,
                        obstacle_prob=0.25,
                        transition_prob=0.8,
                        obs_success_prob=0.9,
                        collision_penalty=1)

    elif env_type == "dynmaze_v1_det":
        size = 9
        s_dims = [size, size]
        moves = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]
        num_action = len(moves)
        stay_action = 4
        k_dims = [3, 3]
        observe_directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        obs_len = len(observe_directions)
        num_obs = 17
        num_outcomes = 17

        env = DynMazeV1(x_size=size,
                        y_size=size,
                        obs_dirs=observe_directions,
                        transition_prob=1.0,
                        obs_success_prob=1.0,
                        collision_penalty=1)

    elif env_type == "dynmaze_v1_nondet":
        size = 9
        s_dims = [size, size]
        moves = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]
        num_action = len(moves)
        stay_action = 4
        k_dims = [3, 3]
        observe_directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        obs_len = len(observe_directions)
        num_obs = 17
        num_outcomes = 17

        env = DynMazeV1(x_size=size,
                        y_size=size,
                        obs_dirs=observe_directions,
                        transition_prob=0.8,
                        obs_success_prob=0.9,
                        collision_penalty=1)

    elif env_type == "dynmaze_v2_det":
        size = 9
        s_dims = [size, size]
        moves = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]
        num_action = len(moves)
        stay_action = 4
        k_dims = [3, 3]
        observe_directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        obs_len = len(observe_directions)
        num_obs = 82
        num_outcomes = 82

        env = DynMazeV2(x_size=size,
                        y_size=size,
                        obs_dirs=observe_directions,
                        transition_prob=1.0,
                        obs_success_prob=1.0,
                        collision_penalty=1)

    elif env_type == "dynmaze_v2_nondet":
        size = 9
        s_dims = [size, size]
        moves = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]
        num_action = len(moves)
        stay_action = 4
        k_dims = [3, 3]
        observe_directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        obs_len = len(observe_directions)
        num_obs = 82
        num_outcomes = 82

        env = DynMazeV2(x_size=size,
                        y_size=size,
                        obs_dirs=observe_directions,
                        transition_prob=0.8,
                        obs_success_prob=0.9,
                        collision_penalty=1)

    elif env_type == "full_obs_grid":
        size = 10
        s_dims = [size, size]
        moves = [[-1, 0], [1, 0], [0, 1], [0, -1], [-1, 1], [-1, -1], [1, 1], [1, -1], [0, 0]]
        num_action = len(moves)
        stay_action = 8
        k_dims = [3, 3]
        obs_len = 2
        num_obs = 100
        num_outcomes = 9

        env = FullObsGrid(x_size=size,
                          y_size=size,
                          obstacle_prob=0.25,
                          transition_prob=0.8,
                          collision_penalty=1)

    elif env_type == "full_obs_grid_20":
        size = 20
        s_dims = [size, size]
        moves = [[-1, 0], [1, 0], [0, 1], [0, -1], [-1, 1], [-1, -1], [1, 1], [1, -1], [0, 0]]
        num_action = len(moves)
        stay_action = 8
        k_dims = [3, 3]
        obs_len = 2
        num_obs = 100
        num_outcomes = 9

        env = FullObsGrid(x_size=size,
                          y_size=size,
                          obstacle_prob=0.25,
                          transition_prob=0.8,
                          collision_penalty=1)

    elif env_type == "full_obs_grid_small":
        size = 6
        s_dims = [size, size]
        moves = [[-1, 0], [1, 0], [0, 1], [0, -1], [-1, 1], [-1, -1], [1, 1], [1, -1], [0, 0]]
        num_action = len(moves)
        stay_action = 8
        k_dims = [3, 3]
        obs_len = 2
        num_obs = 36
        num_outcomes = 9

        env = FullObsGrid(x_size=size,
                          y_size=size,
                          obstacle_prob=0.25,
                          transition_prob=0.8,
                          collision_penalty=1)

    elif env_type == "full_obs_grid_sd":
        size = 10
        s_dims = [size, size]
        moves = [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]]
        num_action = len(moves)
        stay_action = 8
        k_dims = [3, 3]
        obs_len = 2
        num_obs = 100
        num_outcomes = 5

        env = FullObsGrid(x_size=size,
                          y_size=size,
                          obstacle_prob=0.25,
                          transition_prob=0.8,
                          collision_penalty=1)

    elif env_type == "full_obs_grid_sd20":
        size = 20
        s_dims = [size, size]
        moves = [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]]
        num_action = len(moves)
        stay_action = 8
        k_dims = [3, 3]
        obs_len = 2
        num_obs = 100
        num_outcomes = 5

        env = FullObsGrid(x_size=size,
                          y_size=size,
                          obstacle_prob=0.25,
                          transition_prob=0.8,
                          collision_penalty=1)

    elif env_type == "full_obs_grid3d":
        size = 7
        s_dims = [size, size, size]
        moves = [[0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0], [-1, 0, 0], [1, 0, 0], [0, 0, 0]]
        num_action = len(moves)
        stay_action = 6
        k_dims = [3, 3, 3]
        obs_len = 3
        num_obs = 343
        num_outcomes = 7

        env = FullObsGrid3D(x_size=size,
                            y_size=size,
                            z_size=size,
                            obstacle_prob=0.25,
                            transition_prob=0.8,
                            collision_penalty=1)

    elif env_type == "full_obs_dynmaze_v1":
        size = 9
        s_dims = [size, size]
        moves = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]
        num_action = len(moves)
        stay_action = 4
        k_dims = [3, 3]
        obs_len = 2
        num_obs = 17
        num_outcomes = 17

        env = FullObsDynMazeV1(x_size=size,
                               y_size=size,
                               transition_prob=0.8,
                               collision_penalty=1)

    elif env_type == "full_obs_dynmaze_v2":
        size = 9
        s_dims = [size, size]
        moves = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]
        num_action = len(moves)
        stay_action = 4
        k_dims = [3, 3]
        obs_len = 3
        num_obs = 82
        num_outcomes = 82

        env = FullObsDynMazeV2(x_size=size,
                               y_size=size,
                               transition_prob=0.8,
                               collision_penalty=1)

    else:
        print("Environment type not recognised.")
        return

    dims_str = ""
    for i in range(len(s_dims) - 1):
        dims_str += str(s_dims[i])
        dims_str += 'x'
    dims_str += str(s_dims[-1])
    filename = "inputs/env_" + env_type + "_" + dims_str + "_" + suffix + ".txt"

    # create a new file
    f = open(filename, "w")

    # write header information
    f.write("#env_type:'" + env_type + "'\n")
    f.write("#s_dims:" + str(s_dims) + "\n")
    f.write("#num_action:" + str(num_action) + "\n")
    f.write("#stay_action:" + str(stay_action) + "\n")
    f.write("#k_dims:" + str(k_dims) + "\n")
    f.write("#obs_len:" + str(obs_len) + "\n")
    f.write("#num_obs:" + str(num_obs) + "\n")
    f.write("#num_outcomes:" + str(num_outcomes) + "\n")

    # write training data
    generate_input_data(env, num_training_envs, num_trajs_per_env, stay_action, s_dims, f, expert_class)

    f.close()


def generate_input_data(env, max_training_environs, trajectories_per_env, stay_action, s_dims, f, expert_class):

    verbose = True
    debug_print = False

    max_ep_length = max(env.get_state_tensor_shape()) * 3

    step_size = 4
    mutate_prob = env.mutate_prob * step_size

    force_init_belief_during_training = True

    max_traj_attempts = 10

    if type(env) == DynMazeV2:
        limit_obs_length = True
    else:
        limit_obs_length = False
    max_obs_length = 4

    # %%%%% Generate Training Data %%%%%
    # training data set
    if verbose:
        print("=== Generating Training Environments ===")
    f.write("train\n")
    env_count = 0
    # loop over environments
    while env_count < max_training_environs:
        # plan on environment
        expert = expert_class(env.get_state_tensor_shape(),
                              env.get_actions_tensor_shape()[0],
                              env.get_obs_tensor_shape()[0],
                              env.get_transition_model(),
                              env.get_reward_model(),
                              env.get_obs_model(),
                              get_true_state=env.get_true_state)
        if debug_print:
            env.render()
        # generate training trajectories
        attempts = 0
        traj_count = 0
        # loop over trajectories
        while True:
            # initialise trajectory
            traj = []
            b_init = env.get_random_b0()
            # set expert initial belief
            if force_init_belief_during_training:
                expert.force_belief_state(b_init)
                # update expert belief state
                expert.update_belief(stay_action, env.get_init_obs()[0])
            else:
                expert.initialise_belief(env.get_init_obs()[0])
            mutate_steps = []
            # run episode
            last_obs = env.get_init_obs()[0]
            # loop over episode steps
            for j in range(max_ep_length):
                a_tgt = expert.expert_action()
                rewards, observations, done = env.step([a_tgt])
                o = observations[0]

                # store this data point
                traj.append((last_obs, a_tgt))

                last_obs = o

                # if expert reached the goal, store this trajectory
                if done:
                    # if this is the first trajectory, write theta_m to file
                    if traj_count == 0:
                        # write theta_m to file
                        f.write("tm\n")
                        theta_m = env.get_map_params()
                        f.write(format_array(theta_m))
                        f.write("\n")

                        # write theta_m2 to file
                        if mutate_prob > 0:
                            f.write("tm2\n")
                            env.mutate()
                            theta_m2 = env.get_map_params()
                            f.write(format_array(theta_m2))
                            f.write("\n")
                            env.mutate()

                    # write theta_t to file
                    f.write("tt\n")
                    theta_t = env.get_task_params()
                    f.write(format_array(theta_t))
                    f.write("\n")

                    # write b_init to file
                    f.write("bel\n")
                    # limit to length of s_dims
                    b_init_1 = np.array(b_init)
                    while b_init_1.ndim > len(s_dims):
                        b_init_1 = np.sum(b_init_1, axis=-1)
                    f.write(format_array(b_init_1))
                    f.write("\n")

                    f.write("steps\n")
                    for k in range(len(traj)):
                        step = traj[k]
                        obs, a_tgt = step
                        # cut observation to standard size
                        if limit_obs_length:
                            obs = obs[:max_obs_length]
                        # write observation to file
                        line = ""
                        for x in obs:
                            line += str(int(x))
                            line += " "
                        f.write(line + "\n")
                        # write target action to file
                        f.write(str(a_tgt) + "\n")

                        if k in mutate_steps:
                            f.write("mutate\n")

                    traj_count += 1

                    if verbose:
                        print("Trajectory added.")

                    break

                # update expert belief state
                expert.update_belief(a_tgt, o)

                # chance to mutate environment
                if ((j + 1) % step_size == 0) and (random.random() < mutate_prob):
                    if debug_print:
                        env.render()
                        print("Mutating...")
                    if env.mutate():
                        mutate_steps.append(j)

                    # update value table
                    expert.replan(transition_model=env.get_transition_model(),
                                  reward_model=env.get_reward_model(),
                                  observation_model=env.get_obs_model())

                    # update belief based on new observation of environment
                    last_obs = env.get_init_obs()[0]
                    expert.update_belief(stay_action, last_obs)

                if debug_print:
                    env.render()

            attempts += 1

            # if desired trajectories_per_env is reached, generate new environment
            if traj_count >= trajectories_per_env or (traj_count == 0 and attempts > max_traj_attempts):
                break
            # reset for next trajectory
            if verbose:
                print("Regenerating...")
            env.regen_start_and_goal_pos()
            expert.replan(transition_model=env.get_transition_model(),
                          reward_model=env.get_reward_model(),
                          observation_model=env.get_obs_model())

            if debug_print:
                env.render()

        # reset environment
        env.reset()

        if traj_count >= trajectories_per_env:
            env_count += 1

            if verbose:
                print("Env " + str(env_count) + " complete")

    f.write("end\n")


def format_array(x):
    """
    Convert numpy array x to a string which is a valid python array declaration
    :param x: array
    :return: string representing x as an array declaration
    """
    s = "["
    if len(np.shape(x)) == 1:
        # base case
        for i in x:
            s += str(i)
            s += ','
    else:
        # recursive case
        s0 = np.shape(x)[0]
        for i in range(s0):
            s += format_array(x[i])
            s += ','
    s += "]"
    return s


if __name__ == '__main__':
    main(sys.argv[1:])  # skip filename

