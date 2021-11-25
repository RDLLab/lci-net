import numpy as np
import random

# fully observable
from fully_observable.envs.full_obs_grid import FullObsGrid
from fully_observable.envs.full_obs_grid3d import FullObsGrid3D
from fully_observable.envs.full_obs_dynmaze_v1 import FullObsDynMazeV1
from fully_observable.envs.full_obs_dynmaze_v2 import FullObsDynMazeV2
from fully_observable.envs.full_obs_imagegrid import FullObsImageGrid2D
from fully_observable.vi_expert import VIExpert

# partially observable
from partially_obserable.envs.grid2d import Grid2D
from partially_obserable.envs.grid3d import Grid3D
from partially_obserable.envs.dynmaze_v1 import DynMazeV1
from partially_obserable.envs.dynmaze_v2 import DynMazeV2
from partially_obserable.envs.grasper2d import Grasper2D
from partially_obserable.envs.grasper3d import Grasper3D
from partially_obserable.envs.imagegrid import ImageGrid2D
from partially_obserable.qmdp_expert import QMDPExpert


class EvalIterator:
    # no difference to v1 beyond re-naming

    def __init__(self, params, env_type, policy):
        self.params = params
        # adjust values below
        self.max_steps = 300
        self.mutate_prob = 0.125

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
                         # obs_dirs=[(-1, 0), (0, -1), (1, 0), (0, 1)], # old order
                         obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],   # new order
                         obstacle_prob=0.25,
                         transition_prob=0.8,
                         obs_success_prob=0.9,
                         collision_penalty=1)

        elif env_type == "grid_3d_det":
            size = 7
            observe_directions = [[0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
            env = Grid3D(x_size=size,
                         y_size=size,
                         z_size=size,
                         obs_dirs=observe_directions,
                         obstacle_prob=0.25,
                         transition_prob=1.0,
                         obs_success_prob=1.0,
                         collision_penalty=1)

        elif env_type == "grid_3d_nondet":
            size = 7
            observe_directions = [[0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
            env = Grid3D(x_size=size,
                         y_size=size,
                         z_size=size,
                         obs_dirs=observe_directions,
                         obstacle_prob=0.25,
                         transition_prob=0.8,
                         obs_success_prob=0.9,
                         collision_penalty=1)

        elif env_type == "grasper2d_det":
            size = 9
            obj_size = 4
            observe_directions = [[0, -2], [0, 0], [0, 0], [0, 2], [1, -1], [1, 1]]
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
            size = 9
            obj_size = 4
            observe_directions = [[0, -2], [0, 0], [0, 0], [0, 2], [1, -1], [1, 1]]
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
            size = 9
            obj_size = 4
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
            size = 9
            obj_size = 4
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

        elif env_type == "intel_labs_det":
            env = ImageGrid2D(filename="inputs/intel_labs.png",
                              obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                              transition_prob=1.0,
                              obs_success_prob=1.0,
                              collision_penalty=1)
        elif env_type == "intel_labs_nondet":
            env = ImageGrid2D(filename="inputs/intel_labs.png",
                              obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                              transition_prob=0.8,
                              obs_success_prob=0.9,
                              collision_penalty=1)
        elif env_type == "freiburg_det":
            env = ImageGrid2D(filename="inputs/freiburg.png",
                              obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                              transition_prob=1.0,
                              obs_success_prob=1.0,
                              collision_penalty=1)
        elif env_type == "freiburg_nondet":
            env = ImageGrid2D(filename="inputs/freiburg.png",
                              obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                              transition_prob=0.8,
                              obs_success_prob=0.9,
                              collision_penalty=1)
        elif env_type == "hospital_det":
            env = ImageGrid2D(filename="inputs/hospital.png",
                              obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                              transition_prob=1.0,
                              obs_success_prob=1.0,
                              collision_penalty=1)
        elif env_type == "hospital_nondet":
            env = ImageGrid2D(filename="inputs/hospital.png",
                              obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                              transition_prob=0.8,
                              obs_success_prob=0.9,
                              collision_penalty=1)
        elif env_type == "mit_building_det":
            env = ImageGrid2D(filename="inputs/mit_building.png",
                              obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                              transition_prob=1.0,
                              obs_success_prob=1.0,
                              collision_penalty=1)
        elif env_type == "mit_building_nondet":
            env = ImageGrid2D(filename="inputs/mit_building.png",
                              obs_dirs=[(0, 1), (1, 0), (0, -1), (-1, 0)],
                              transition_prob=0.8,
                              obs_success_prob=0.9,
                              collision_penalty=1)
        elif env_type == "full_obs_grid":
            size = 10
            env = FullObsGrid(x_size=size,
                              y_size=size,
                              obstacle_prob=0.25,
                              transition_prob=0.8,
                              collision_penalty=1)
        elif env_type == "full_obs_grid_small":
            size = 6
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
        self.policy = policy
        if params.include_expert:
            if 'full_obs' in env_type:
                expert_class = VIExpert
            else:
                expert_class = QMDPExpert
            self.expert = expert_class(env.get_state_tensor_shape(),
                                       env.get_actions_tensor_shape()[0],
                                       env.get_obs_tensor_shape()[0],
                                       env.get_transition_model(),
                                       env.get_reward_model(),
                                       env.get_obs_model(),
                                       get_true_state=env.get_true_state)
        else:
            self.expert = None

        self.trial = 0
        self.traj = 0

    def next(self):
        # simulate a trajectory for both expert and network
        mut_steps = []
        b_init = self.env.get_random_b0()

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
                    collisions += 1
                if done:
                    success = 1
                    break

                # chance to mutate env
                if random.random() < self.mutate_prob:
                    if self.env.mutate():
                        mut_steps.append(i)
                        self.expert.replan(transition_model=self.env.get_transition_model(),
                                           observation_model=self.env.get_obs_model())

            collision_rate = 1.0 * collisions / traj_len
            res_exp = [success, traj_len, collision_rate, reward]
            self.env.reset_to_start()
        else:
            res_exp = [-1, -1, -1, -1]  # not used

        # %%%%% agent %%%%%
        success = 0
        traj_len = 0
        collisions = 0
        reward = 0
        # cut belief to correct dimension (required for dynmaze v2 only)
        if len(self.params.s_dims) == 2 and b_init.ndim == 3:
            b_init = b_init[:,:,0]

        self.policy.reset(self.env.get_map_params(), self.env.get_task_params(), b_init)
        last_a = self.params.stay_action
        last_o = self.env.get_init_obs()[0]

        for i in range(self.max_steps):
            # cut obs to correct dimension (required for dynmaze v2 only)
            if len(last_o) == 5 and self.params.obs_len == 4:
                last_o = last_o[:4]

            a = self.policy.eval(last_a, last_o)
            r, o, done = self.env.step([a])
            last_a = a
            last_o = o[0]

            # update traj results
            traj_len += 1
            if r[0] < -0.001:
                collisions += 1
            if done:
                success = 1
                break

            # mutate on same steps as expert
            if i in mut_steps:
                if self.env.mutate():
                    # partial reset of policy
                    self.policy.env_img = np.reshape(self.env.get_map_params(),
                                                     [1] + self.params.s_dims)
                    last_o = self.env.get_init_obs()[0]

        collision_rate = 1.0 * collisions / traj_len
        res_net = [success, traj_len, collision_rate, reward]
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

        # return results for this trajectory
        return [res_exp, res_net]

