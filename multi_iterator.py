import numpy as np
import random

from PIL import Image, ImageDraw, ImageFont

# fully observable
from fully_observable.envs.full_obs_grid import FullObsGrid
from fully_observable.envs.full_obs_grid3d import FullObsGrid3D
from fully_observable.envs.full_obs_dynmaze_v1 import FullObsDynMazeV1
from fully_observable.envs.full_obs_dynmaze_v2 import FullObsDynMazeV2
from fully_observable.envs.full_obs_imagegrid import FullObsImageGrid2D

# partially observable
from partially_obserable.qmdp_expert import QMDPExpert
from partially_obserable.envs.grid2d import Grid2D
from partially_obserable.envs.grid3d import Grid3D
from partially_obserable.envs.dynmaze_v1 import DynMazeV1
from partially_obserable.envs.dynmaze_v2 import DynMazeV2
from partially_obserable.envs.grasper2d import Grasper2D
from partially_obserable.envs.grasper3d import Grasper3D
from partially_obserable.envs.imagegrid import ImageGrid2D


class MultiIterator:

    def __init__(self, params, env_type, policies, record=False):
        self.params = params
        # adjust values below
        self.max_steps = 80     # overridden below for large envs
        self.mutate_prob = 0.125

        self.record = record

        if env_type == "grid_det":
            size = 10
            env = Grid2D(x_size=size,
                         y_size=size,
                         obs_dirs=[(-1, 0), (0, -1), (1, 0), (0, 1)],
                         obstacle_prob=0.25,
                         transition_prob=1.0,
                         obs_success_prob=1.0,
                         collision_penalty=10)

        elif env_type == "grid_det_large":
            size = 20
            env = Grid2D(x_size=size,
                         y_size=size,
                         obs_dirs=[(-1, 0), (0, -1), (1, 0), (0, 1)],
                         obstacle_prob=0.25,
                         transition_prob=1.0,
                         obs_success_prob=1.0,
                         collision_penalty=10)

        elif env_type == "grid_nondet":
            size = 10
            env = Grid2D(x_size=size,
                         y_size=size,
                         obs_dirs=[(-1, 0), (0, -1), (1, 0), (0, 1)],
                         obstacle_prob=0.25,
                         transition_prob=0.8,
                         obs_success_prob=0.9,
                         collision_penalty=1)

        elif env_type == "grid_nondet_large":
            size = 20
            env = Grid2D(x_size=size,
                         y_size=size,
                         obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                         obstacle_prob=0.25,
                         transition_prob=0.8,
                         obs_success_prob=0.9,
                         collision_penalty=1)

        elif env_type == "grid_3d_det":
            size = 7
            env = Grid3D(x_size=size,
                         y_size=size,
                         z_size=size,
                         obs_dirs=[[0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0], [1, 0, 0], [-1, 0, 0]],
                         obstacle_prob=0.25,
                         transition_prob=1.0,
                         obs_success_prob=1.0,
                         collision_penalty=1)

        elif env_type == "grid_3d_nondet":
            size = 7
            env = Grid3D(x_size=size,
                         y_size=size,
                         z_size=size,
                         obs_dirs=[[0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0], [1, 0, 0], [-1, 0, 0]],
                         obstacle_prob=0.25,
                         transition_prob=0.8,
                         obs_success_prob=0.9,
                         collision_penalty=1)

        elif env_type == "grid_3d_det_large":
            size = 14
            env = Grid3D(x_size=size,
                         y_size=size,
                         z_size=size,
                         obs_dirs=[[0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0], [1, 0, 0], [-1, 0, 0]],
                         obstacle_prob=0.25,
                         transition_prob=1.0,
                         obs_success_prob=1.0,
                         collision_penalty=1)
            self.max_steps = 50

        elif env_type == "grid_3d_nondet_large":
            size = 14
            env = Grid3D(x_size=size,
                         y_size=size,
                         z_size=size,
                         obs_dirs=[[0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0], [1, 0, 0], [-1, 0, 0]],
                         obstacle_prob=0.25,
                         transition_prob=0.8,
                         obs_success_prob=0.9,
                         collision_penalty=1)
            self.max_steps = 50

        elif env_type == "grasper2d_det":
            size = 14
            obj_size = 6
            env = Grasper2D(x_size=size,
                            y_size=size,
                            obj_max_x=obj_size,
                            obj_max_y=obj_size,
                            obs_dirs=[[0, -2], [0, 0], [0, 0], [0, 2], [1, -1], [1, 1]],
                            obstacle_prob=0.25,
                            transition_prob=1.0,
                            obs_success_prob=1.0,
                            collision_penalty=1)

        elif env_type == "grasper2d_nondet":
            size = 14
            obj_size = 6
            env = Grasper2D(x_size=size,
                            y_size=size,
                            obj_max_x=obj_size,
                            obj_max_y=obj_size,
                            obs_dirs=[[0, -2], [0, 0], [0, 0], [0, 2], [1, -1], [1, 1]],
                            obstacle_prob=0.25,
                            transition_prob=0.8,
                            obs_success_prob=0.9,
                            collision_penalty=1)

        elif env_type == "grasper3d_det":
            size = 7
            obj_size = 3
            env = Grasper3D(x_size=size,
                            y_size=size,
                            z_size=size,
                            obj_max_x=obj_size,
                            obj_max_y=obj_size,
                            obj_max_z=obj_size,
                            obs_dirs=[[0, 0, -2], [0, -1, -1], [0, 1, -1], [0, 0, 0], [0, 0, 2], [0, -1, 1], [0, 1, 1]],
                            obstacle_prob=0.25,
                            transition_prob=1.0,
                            obs_success_prob=1.0,
                            collision_penalty=1)

        elif env_type == "grasper3d_nondet":
            size = 7
            obj_size = 3
            env = Grasper3D(x_size=size,
                            y_size=size,
                            z_size=size,
                            obj_max_x=obj_size,
                            obj_max_y=obj_size,
                            obj_max_z=obj_size,
                            obs_dirs=[[0, 0, -2], [0, -1, -1], [0, 1, -1], [0, 0, 0], [0, 0, 2], [0, -1, 1], [0, 1, 1]],
                            obstacle_prob=0.25,
                            transition_prob=0.8,
                            obs_success_prob=0.9,
                            collision_penalty=1)

        elif env_type == "grasper3d_det_large":
            size = 14
            obj_size = 6
            env = Grasper3D(x_size=size,
                            y_size=size,
                            z_size=size,
                            obj_max_x=obj_size,
                            obj_max_y=obj_size,
                            obj_max_z=obj_size,
                            obs_dirs=[[0, 0, -2], [0, -1, -1], [0, 1, -1], [0, 0, 0], [0, 0, 2], [0, -1, 1], [0, 1, 1]],
                            obstacle_prob=0.25,
                            transition_prob=1.0,
                            obs_success_prob=1.0,
                            collision_penalty=1)

        elif env_type == "grasper3d_nondet_large":
            size = 14
            obj_size = 6
            env = Grasper3D(x_size=size,
                            y_size=size,
                            z_size=size,
                            obj_max_x=obj_size,
                            obj_max_y=obj_size,
                            obj_max_z=obj_size,
                            obs_dirs=[[0, 0, -2], [0, -1, -1], [0, 1, -1], [0, 0, 0], [0, 0, 2], [0, -1, 1], [0, 1, 1]],
                            obstacle_prob=0.25,
                            transition_prob=0.8,
                            obs_success_prob=0.9,
                            collision_penalty=1)

        elif env_type == "dynmaze_v1_det":
            size = 9
            env = DynMazeV1(x_size=size,
                            y_size=size,
                            obs_dirs=[(-1, 0), (0, -1), (1, 0), (0, 1)],
                            transition_prob=1.0,
                            obs_success_prob=1.0,
                            collision_penalty=1)

        elif env_type == "dynmaze_v1_det_large":
            size = 29
            env = DynMazeV1(x_size=size,
                            y_size=size,
                            obs_dirs=[(-1, 0), (0, -1), (1, 0), (0, 1)],
                            transition_prob=1.0,
                            obs_success_prob=1.0,
                            collision_penalty=1)
            self.max_steps = 160

        elif env_type == "dynmaze_v1_nondet":
            size = 9
            env = DynMazeV1(x_size=size,
                            y_size=size,
                            obs_dirs=[(-1, 0), (0, -1), (1, 0), (0, 1)],
                            transition_prob=0.8,
                            obs_success_prob=0.9,
                            collision_penalty=1)

        elif env_type == "dynmaze_v1_nondet_large":
            size = 29
            env = DynMazeV1(x_size=size,
                            y_size=size,
                            obs_dirs=[(-1, 0), (0, -1), (1, 0), (0, 1)],
                            transition_prob=0.8,
                            obs_success_prob=0.9,
                            collision_penalty=1)
            self.max_steps = 160

        elif env_type == "dynmaze_v2_det":
            size = 9
            env = DynMazeV2(x_size=size,
                            y_size=size,
                            obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                            transition_prob=1.0,
                            obs_success_prob=1.0,
                            collision_penalty=1)

        elif env_type == "dynmaze_v2_det_large":
            size = 29
            env = DynMazeV2(x_size=size,
                            y_size=size,
                            obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                            transition_prob=1.0,
                            obs_success_prob=1.0,
                            collision_penalty=1)
            self.max_steps = 160

        elif env_type == "dynmaze_v2_nondet":
            size = 9
            env = DynMazeV2(x_size=size,
                            y_size=size,
                            obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                            transition_prob=0.8,
                            obs_success_prob=0.9,
                            collision_penalty=1)

        elif env_type == "dynmaze_v2_nondet_large":
            size = 29
            env = DynMazeV2(x_size=size,
                            y_size=size,
                            obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                            transition_prob=0.8,
                            obs_success_prob=0.9,
                            collision_penalty=1)
            self.max_steps = 160

        elif env_type == "intel_labs_det":
            env = ImageGrid2D(filename="inputs/intel_labs.png",
                              obs_dirs=[(-1, 0), (0, -1), (1, 0), (0, 1)],
                              transition_prob=1.0,
                              obs_success_prob=1.0,
                              collision_penalty=1)
            self.max_steps = 300

        elif env_type == "intel_labs_nondet":
            env = ImageGrid2D(filename="inputs/intel_labs.png",
                              obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                              transition_prob=0.8,
                              obs_success_prob=0.9,
                              collision_penalty=1)
            self.max_steps = 300

        elif env_type == "freiburg_det":
            env = ImageGrid2D(filename="inputs/freiburg.png",
                              obs_dirs=[(-1, 0), (0, -1), (1, 0), (0, 1)],
                              transition_prob=1.0,
                              obs_success_prob=1.0,
                              collision_penalty=1)
            self.max_steps = 300

        elif env_type == "freiburg_nondet":
            env = ImageGrid2D(filename="inputs/freiburg.png",
                              obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                              transition_prob=0.8,
                              obs_success_prob=0.9,
                              collision_penalty=1)
            self.max_steps = 300

        elif env_type == "hospital_det":
            env = ImageGrid2D(filename="inputs/hospital.png",
                              obs_dirs=[(-1, 0), (0, -1), (1, 0), (0, 1)],
                              transition_prob=1.0,
                              obs_success_prob=1.0,
                              collision_penalty=1)
            self.max_steps = 300

        elif env_type == "hospital_nondet":
            env = ImageGrid2D(filename="inputs/hospital.png",
                              obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                              transition_prob=0.8,
                              obs_success_prob=0.9,
                              collision_penalty=1)
            self.max_steps = 300

        elif env_type == "mit_building_det":
            env = ImageGrid2D(filename="inputs/mit_building.png",
                              obs_dirs=[(-1, 0), (0, -1), (1, 0), (0, 1)],
                              transition_prob=1.0,
                              obs_success_prob=1.0,
                              collision_penalty=1)
            self.max_steps = 400

        elif env_type == "mit_building_nondet":
            env = ImageGrid2D(filename="inputs/mit_building.png",
                              obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                              transition_prob=0.8,
                              obs_success_prob=0.9,
                              collision_penalty=1)
            self.max_steps = 400

        elif env_type == "full_obs_grid":
            size = 10
            env = FullObsGrid(x_size=size,
                              y_size=size,
                              obstacle_prob=0.25,
                              transition_prob=0.8,
                              collision_penalty=1)

        elif env_type == "full_obs_grid3d":
            size = 7
            env = FullObsGrid3D(x_size=size,
                                y_size=size,
                                z_size=size,
                                obstacle_prob=0.25,
                                transition_prob=0.8,
                                collision_penalty=1)

        elif env_type == "full_obs_dynmaze_v1":
            size = 9
            env = FullObsDynMazeV1(x_size=size,
                                   y_size=size,
                                   transition_prob=0.8,
                                   collision_penalty=1)

        elif env_type == "full_obs_dynmaze_v2":
            size = 9
            env = FullObsDynMazeV2(x_size=size,
                                   y_size=size,
                                   transition_prob=0.8,
                                   collision_penalty=1)

        elif env_type == "full_obs_grid3d_large":
            size = 14
            env = FullObsGrid3D(x_size=size,
                                y_size=size,
                                z_size=size,
                                obstacle_prob=0.25,
                                transition_prob=0.8,
                                collision_penalty=1)

        elif env_type == "full_obs_dynmaze_v1_large":
            size = 29
            env = FullObsDynMazeV1(x_size=size,
                                   y_size=size,
                                   transition_prob=0.8,
                                   collision_penalty=1)

        elif env_type == "full_obs_dynmaze_v2_large":
            size = 29
            env = FullObsDynMazeV2(x_size=size,
                                   y_size=size,
                                   transition_prob=0.8,
                                   collision_penalty=1)

        elif env_type == "full_obs_intel_labs_det":
            env = FullObsImageGrid2D(filename="inputs/intel_labs.png",
                                     transition_prob=1.0,
                                     collision_penalty=1)
            self.max_steps = 300

        elif env_type == "full_obs_intel_labs_nondet":
            env = FullObsImageGrid2D(filename="inputs/intel_labs.png",
                                     transition_prob=0.8,
                                     collision_penalty=1)
            self.max_steps = 300

        elif env_type == "full_obs_freiburg_det":
            env = FullObsImageGrid2D(filename="inputs/freiburg.png",
                                     transition_prob=1.0,
                                     collision_penalty=1)
            self.max_steps = 300

        elif env_type == "full_obs_freiburg_nondet":
            env = FullObsImageGrid2D(filename="inputs/freiburg.png",
                                     transition_prob=0.8,
                                     collision_penalty=1)
            self.max_steps = 300

        elif env_type == "full_obs_hospital_det":
            env = FullObsImageGrid2D(filename="inputs/hospital.png",
                                     transition_prob=1.0,
                                     collision_penalty=1)
            self.max_steps = 300

        elif env_type == "full_obs_hospital_nondet":
            env = FullObsImageGrid2D(filename="inputs/hospital.png",
                                     transition_prob=0.8,
                                     collision_penalty=1)
            self.max_steps = 300

        elif env_type == "full_obs_mit_building_det":
            env = FullObsImageGrid2D(filename="inputs/mit_building.png",
                                     transition_prob=1.0,
                                     collision_penalty=1)
            self.max_steps = 400

        elif env_type == "full_obs_mit_building_nondet":
            env = FullObsImageGrid2D(filename="inputs/mit_building.png",
                                     transition_prob=0.8,
                                     collision_penalty=1)
            self.max_steps = 400

        else:
            raise ValueError('Invalid environment type in input file.')

        self.env = env
        self.policies = policies
        if params.include_expert:
            self.expert = QMDPExpert(env.get_state_tensor_shape(),
                                     env.get_actions_tensor_shape()[0],
                                     env.get_obs_tensor_shape()[0],
                                     env.get_transition_model(),
                                     env.get_reward_model(),
                                     env.get_obs_model())
        else:
            self.expert = None

        self.trial = 0
        self.traj = 0

    def next(self):
        # simulate a trajectory for both expert and network
        mut_steps = []
        b_init = self.env.get_random_b0()

        # !!! for record only
        if self.record:
            r_beliefs = [[] for _ in self.policies]
            r_positions = [[] for _ in self.policies]
            r_actions = [[] for _ in self.policies]
            r_map = [[] for _ in self.policies]
            r_goal = None
        # !!! end for record only

        if self.mutate_prob > 0:
            for i in range(self.max_steps):
                # chance to mutate env
                if random.random() < self.mutate_prob:
                    if self.env.mutate():
                        mut_steps.append(i)
        self.env.reset_to_start()

        # %%%%% expert %%%%%
        if self.params.include_expert:
            success = 0
            traj_len = 0
            collisions = 0
            reward = 0
            self.expert.replan(transition_model=self.env.get_transition_model(),
                               reward_model=self.env.get_reward_model(),
                               observation_model=self.env.get_obs_model())
            self.expert.force_belief_state(b_init)
            self.expert.update_belief(self.params.stay_action, self.env.get_init_obs()[0])
            for i in range(self.max_steps):
                a = self.expert.expert_action()
                r, o, done = self.env.step([a])
                self.expert.update_belief(a, o[0])

                # update traj results
                traj_len += 1
                if r[0] < -0.001:
                    collided = 1
                    collisions += 1
                if done:
                    success = 1
                    break

                # chance to mutate env
                if i in mut_steps:
                    if self.env.mutate():
                        self.expert.replan(transition_model=self.env.get_transition_model(),
                                           observation_model=self.env.get_obs_model())

            collision_rate = 1.0 * collisions / traj_len
            res_exp = [success, traj_len, collision_rate, reward]
            self.env.reset_to_start()
        else:
            res_exp = [-1, -1, -1, -1]  # not used

        # %%%%% agent %%%%%
        res_all_nets = []
        for (pi, policy) in enumerate(self.policies):
            success = 0
            traj_len = 0
            collided = 0
            collisions = 0
            reward = 0
            # cut belief to correct dimension (required for dynmaze v2 only)
            if len(self.params.s_dims) == 2 and b_init.ndim == 3:
                b_init = b_init[:, :, 0]

            policy.reset(self.env.get_map_params(), self.env.get_task_params(), b_init)
            last_a = self.params.stay_action
            last_o = self.env.get_init_obs()[0]

            # !!! for record only
            if self.record:
                if pi == 0:
                    r_goal = policy.goal_img
                r_map[pi].append(policy.env_img)
                r_beliefs[pi].append(policy.belief_img)
                r_positions[pi].append((self.env.agent_y, self.env.agent_x))    # record only supports 2D
            # !!! end for record only

            for i in range(self.max_steps):
                # cut obs to correct dimension (required for dynmaze v2 only)
                if len(last_o) == 5 and self.params.obs_len == 4:
                    last_o = last_o[:4]

                a = policy.eval(last_a, last_o)
                r, o, done = self.env.step([a])
                last_a = a
                last_o = o[0]

                # !!! for record only
                if self.record:
                    r_map[pi].append(policy.env_img)
                    r_beliefs[pi].append(policy.sess.run(policy.network.belief))
                    r_positions[pi].append((self.env.agent_y, self.env.agent_x))
                    r_actions[pi].append(a)
                # !!! end for record only

                # update traj results
                traj_len += 1
                if r[0] < -0.001:
                    collided = 1
                    collisions += 1
                if done:
                    success = 1
                    break

                # mutate on same steps as expert
                if i in mut_steps:
                    if self.env.mutate():
                        # partial reset of policy
                        policy.env_img = np.reshape(self.env.get_map_params(),
                                                    [1] + self.params.s_dims)
                        last_o = self.env.get_init_obs()[0]

            collision_rate = 1.0 * collisions / traj_len
            # res_net = [success, traj_len, collided, reward]
            res_net = [success, traj_len, collision_rate, reward]
            res_all_nets.append(res_net)
            self.env.reset_to_start()
        self.trial += 1

        # go to next traj if necessary
        if self.trial >= self.params.eval_repeats:
            self.trial = 0
            self.traj += 1

            # go to next env if necessary
            if self.traj >= self.params.eval_trajs:
                self.traj = 0
                self.env.reset()
                if self.params.include_expert:
                    # do full re-plan
                    self.expert.replan(transition_model=self.env.get_transition_model(),
                                       reward_model=self.env.get_reward_model(),
                                       observation_model=self.env.get_obs_model())

            else:
                self.env.regen_start_and_goal_pos()
                if self.params.include_expert:
                    # re-plan goal only
                    self.expert.replan(reward_model=self.env.get_reward_model())

        # !!! for record only
        if self.record:
            # visualise_path(r_map, r_goal, r_beliefs, r_positions, r_actions)
            animate_path(r_map, r_goal, r_beliefs, r_positions, r_actions)
        # !!! end for record only

        # return results for this trajectory
        return [res_exp] + res_all_nets


def visualise_path(env_map, goal, beliefs, positions, actions):
    prefix = 'outputs/path_vis_'
    suffix = '.png'

    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    grey = (128, 128, 128)

    scale_factor = 25
    line_scale_factor = 3

    offset_base = 5
    offset_tnx = 15

    env_map = env_map[0]
    n, m = np.shape(env_map)
    goal = goal[0]

    beliefs_base = beliefs[0][:][0][:][:]
    beliefs_tnx = beliefs[1][:][0][:][:]
    positions_base = positions[0]
    positions_tnx = positions[1]
    actions_base = actions[0]
    actions_tnx = actions[1]

    map_img = Image.new('RGB', (n * scale_factor, m * scale_factor), 'white')
    pixels = map_img.load()

    # draw obstacles and goal
    for i in range(n):
        for j in range(m):
            if env_map[i][j] == 1:
                # draw a black block for obstacle
                for p in range(scale_factor):
                    for q in range(scale_factor):
                        pixels[(i*scale_factor)+p, (j*scale_factor)+q] = black
            if env_map[i][j] == 2:
                # draw a grey block for gate
                for p in range(scale_factor):
                    for q in range(scale_factor):
                        pixels[(i*scale_factor)+p, (j*scale_factor)+q] = grey
            if goal[i][j] == 1:
                # draw a green block for goal
                for p in range(scale_factor):
                    for q in range(scale_factor):
                        pixels[(i*scale_factor)+p, (j*scale_factor)+q] = green

    # draw both trajectories
    for (positions, offset, colour) in [(positions_base, offset_base, red), (positions_tnx, offset_tnx, blue)]:
        for i in range(len(positions)):
            px, py = positions[i]
            # draw current point
            pixels[(px * scale_factor) + offset, (py * scale_factor) + offset] = colour

            if i == 0 or i == len(positions) - 1:
                for j in range(-(line_scale_factor // 2) - 2, (line_scale_factor // 2) + 3):
                    for k in range(-(line_scale_factor // 2) - 2, (line_scale_factor // 2) + 3):
                        pixels[(px * scale_factor) + offset + j, (py * scale_factor) + offset + k] = colour
            else:
                for j in range(-(line_scale_factor // 2), (line_scale_factor // 2) + 1):
                    for k in range(-(line_scale_factor // 2), (line_scale_factor // 2) + 1):
                        pixels[(px * scale_factor) + offset + j, (py * scale_factor) + offset + k] = colour

            if i >= len(positions) - 1:
                break
            # draw line to next point
            px1, py1 = positions[i + 1]
            if px < px1:
                for j in range(scale_factor):
                    for k in range(-(line_scale_factor // 2), (line_scale_factor // 2) + 1):
                        pixels[(px * scale_factor) + offset + j, (py * scale_factor) + offset + k] = colour
            elif px > px1:
                for j in range(scale_factor):
                    for k in range(-(line_scale_factor // 2), (line_scale_factor // 2) + 1):
                        pixels[(px * scale_factor) + offset - j, (py * scale_factor) + offset + k] = colour
            elif py < py1:
                for j in range(scale_factor):
                    for k in range(-(line_scale_factor // 2), (line_scale_factor // 2) + 1):
                        pixels[(px * scale_factor) + offset + k, (py * scale_factor) + offset + j] = colour
            elif py > py1:
                for j in range(scale_factor):
                    for k in range(-(line_scale_factor // 2), (line_scale_factor // 2) + 1):
                        pixels[(px * scale_factor) + offset + k, (py * scale_factor) + offset - j] = colour

    map_img.show()

    map_img.save(prefix + 'scratch_3' + suffix)

    pass


def animate_path(env_map, goal, beliefs, positions, actions):
    prefix = 'outputs/path_vis_'
    suffix = '.gif'

    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    grey = (128, 128, 128)
    purple = (128, 0, 128)

    scale_factor = 15
    line_scale_factor = 3
    top_border = 40
    bottom_border = 80

    title_msg = "Comparison of Learned Policies on 29x29 Dynamic Maze V2 Example"
    base_msg = "QMDP-Net"
    tnx_msg = "LCI-Net"
    legend_top_msg = "Green indicates goal cell. Belief for each cell is indicated by shade from white to purple - " \
                     "closer to purple indicates stronger belief."
    legend_bot_msg = "Grey indicates the currently closed gate, which changes over time. Closed red or blue square " \
                     "is starting position, open square is current position."

    env_map_base = np.array(env_map[0])[:, 0, :, :]
    env_map_tnx = np.array(env_map[1])[:, 0, :, :]
    if np.shape(env_map_base)[0] >= np.shape(env_map_tnx)[0]:
        env_map = env_map_base
    else:
        env_map = env_map_tnx
    _, n, m = np.shape(env_map)
    goal = goal[0]

    offset_bx = scale_factor // 2
    offset_by = (scale_factor // 2) + top_border
    offset_tx = (scale_factor // 2) + (n * scale_factor)
    offset_ty = (scale_factor // 2) + top_border

    beliefs_base = np.array(beliefs[0])[:, 0, :, :]
    beliefs_tnx = np.array(beliefs[1])[:, 0, :, :]
    positions_base = positions[0]
    positions_tnx = positions[1]
    actions_base = actions[0]
    actions_tnx = actions[1]

    frame_list = []

    # animate trajectories
    for c in range(max(len(positions_base), len(positions_tnx))):
        frame = Image.new('RGB', (n * scale_factor * 2, (m * scale_factor) + top_border + bottom_border), 'white')
        pixels = frame.load()

        # %%% draw title and captions %%%
        fnt_lrg = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", 18)
        fnt_med = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", 14)
        fnt_sml = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 12)
        d = ImageDraw.Draw(frame)
        d.rectangle([(0, 0),
                     (n * scale_factor * 2, top_border)],
                    fill=black)
        d.rectangle([(0,
                      (m * scale_factor) + top_border),
                     (n * scale_factor * 2, (m * scale_factor) + top_border + bottom_border)],
                    fill=black)
        # main title
        tw, th = d.textsize(title_msg, font=fnt_lrg)
        d.text(((n * scale_factor) - (tw // 2),
                (top_border - th + scale_factor) // 2),
               title_msg, font=fnt_lrg, fill=white)

        # base label
        tw, th = d.textsize(base_msg, font=fnt_med)
        d.text((((n * scale_factor - tw) // 2),
                (m * scale_factor) + top_border + (bottom_border // 4) - (th // 2) - scale_factor),
               base_msg, font=fnt_med, fill=white)

        # tnx label
        tw, th = d.textsize(tnx_msg, font=fnt_med)
        d.text((((n * scale_factor - tw) // 2) + (n * scale_factor),
                (m * scale_factor) + top_border + (bottom_border // 4) - (th // 2) - scale_factor),
               tnx_msg, font=fnt_med, fill=white)

        # time step
        time_step_msg = "t = " + str(c)
        tw, th = d.textsize(time_step_msg, font=fnt_lrg)
        d.text(((n * scale_factor) - (tw // 2),
                (m * scale_factor) + top_border + (bottom_border // 4) - (th // 2) - scale_factor),
               time_step_msg, font=fnt_lrg, fill=white)

        # legend
        tw, th = d.textsize(legend_top_msg, font=fnt_sml)
        d.text(((n * scale_factor) - (tw // 2),
                (m * scale_factor) + top_border + (3 * bottom_border // 4) - (th // 2) - scale_factor),
               legend_top_msg, font=fnt_sml, fill=white)
        tw, th = d.textsize(legend_bot_msg, font=fnt_sml)
        d.text(((n * scale_factor) - (tw // 2),
                (m * scale_factor) + top_border + (3 * bottom_border // 4) - (th // 2)),
               legend_bot_msg, font=fnt_sml, fill=white)

        # %%% draw belief images %%%
        b_base_max = np.max(beliefs_base[min(c, len(positions_base) - 1)])
        b_tnx_max = np.max(beliefs_tnx[min(c, len(positions_tnx) - 1)])
        for i in range(n):
            for j in range(m):
                # shade this block based on belief
                b_shade = to_heatmap(white, purple, beliefs_base[min(c, len(positions_base) - 1)][i][j], b_base_max)
                t_shade = to_heatmap(white, purple, beliefs_tnx[min(c, len(positions_tnx) - 1)][i][j], b_tnx_max)
                for p in range(2, scale_factor - 2):
                    for q in range(2, scale_factor - 2):
                        pixels[(j * scale_factor) + p, (i * scale_factor) + q + top_border] = b_shade
                        pixels[(j * scale_factor) + p + (scale_factor * n), (i * scale_factor) + q + top_border] = t_shade

        # %%% draw obstacles and goal %%%
        for i in range(n):
            for j in range(m):
                if env_map[c][i][j] == 1:
                    # draw a black block for obstacle
                    for p in range(scale_factor):
                        for q in range(scale_factor):
                            pixels[(j * scale_factor) + p, (i * scale_factor) + q + top_border] = black
                            pixels[(j * scale_factor) + p + (scale_factor * n), (i * scale_factor) + q + top_border] = black
                if env_map[c][i][j] == 2:
                    # draw a grey block for gate
                    for p in range(scale_factor):
                        for q in range(scale_factor):
                            pixels[(j * scale_factor) + p, (i * scale_factor) + q + top_border] = grey
                            pixels[(j * scale_factor) + p + (scale_factor * n), (i * scale_factor) + q + top_border] = grey
                if goal[i][j] == 1:
                    # draw a green outline for goal
                    for p in range(scale_factor):
                        for q in range(scale_factor):
                            pixels[(j * scale_factor) + p, (i * scale_factor) + q + top_border] = green
                            pixels[(j * scale_factor) + p + (scale_factor * n), (i * scale_factor) + q + top_border] = green

        # %%% draw trajectory up to current step %%%
        for i in range(c):
            for (positions, offset_x, offset_y, colour) in [(positions_base, offset_bx, offset_by, red),
                                                            (positions_tnx, offset_tx, offset_ty, blue)]:
                if i >= len(positions):
                    py, px = positions[-1]
                else:
                    py, px = positions[i]
                # draw current point
                pixels[(px * scale_factor) + offset_x, (py * scale_factor) + offset_y] = colour

                # if i == 0 or i >= len(positions) - 1:
                if i == 0:
                    for j in range(-(line_scale_factor // 2) - 2, (line_scale_factor // 2) + 3):
                        for k in range(-(line_scale_factor // 2) - 2, (line_scale_factor // 2) + 3):
                            pixels[(px * scale_factor) + offset_x + j, (py * scale_factor) + offset_y + k] = colour
                else:
                    for j in range(-(line_scale_factor // 2), (line_scale_factor // 2) + 1):
                        for k in range(-(line_scale_factor // 2), (line_scale_factor // 2) + 1):
                            pixels[(px * scale_factor) + offset_x + j, (py * scale_factor) + offset_y + k] = colour

                if i == c - 1:
                    # highlight current end of trajectory
                    for j in range(-(line_scale_factor // 2) - 2, (line_scale_factor // 2) + 3):
                        for k in range(-(line_scale_factor // 2) - 2, (line_scale_factor // 2) + 3):
                            pixels[(px * scale_factor) + offset_x + j, (py * scale_factor) + offset_y + k] = colour

                    # white centre indicates current position
                    for j in range(-(line_scale_factor // 2), (line_scale_factor // 2) + 1):
                        for k in range(-(line_scale_factor // 2), (line_scale_factor // 2) + 1):
                            pixels[(px * scale_factor) + offset_x + j, (py * scale_factor) + offset_y + k] = white

                    # skip drawing line to next point
                    continue
                # draw line to next point
                if i >= len(positions) - 1:
                    py1, px1 = positions[-1]
                else:
                    py1, px1 = positions[i + 1]
                if px < px1:
                    for j in range(scale_factor):
                        for k in range(-(line_scale_factor // 2), (line_scale_factor // 2) + 1):
                            pixels[(px * scale_factor) + offset_x + j, (py * scale_factor) + offset_y + k] = colour
                elif px > px1:
                    for j in range(scale_factor):
                        for k in range(-(line_scale_factor // 2), (line_scale_factor // 2) + 1):
                            pixels[(px * scale_factor) + offset_x - j, (py * scale_factor) + offset_y + k] = colour
                elif py < py1:
                    for j in range(scale_factor):
                        for k in range(-(line_scale_factor // 2), (line_scale_factor // 2) + 1):
                            pixels[(px * scale_factor) + offset_x + k, (py * scale_factor) + offset_y + j] = colour
                elif py > py1:
                    for j in range(scale_factor):
                        for k in range(-(line_scale_factor // 2), (line_scale_factor // 2) + 1):
                            pixels[(px * scale_factor) + offset_x + k, (py * scale_factor) + offset_y - j] = colour

        frame_list.append(frame)

    frame_list[0].save(prefix + 'video_v3e6' + suffix, save_all=True, append_images=frame_list[1:],
                       optimize=False, duration=200, loop=0)

    pass


def to_heatmap(cold_colour, hot_colour, val, max_val):
    diffs = (hot_colour[0] - cold_colour[0], hot_colour[1] - cold_colour[1], hot_colour[2] - cold_colour[2])
    return (int(cold_colour[0] + (diffs[0] * (val / max_val))),
            int(cold_colour[1] + (diffs[1] * (val / max_val))),
            int(cold_colour[2] + (diffs[2] * (val / max_val))))


