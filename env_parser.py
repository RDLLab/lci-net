import numpy as np
import random

from train_iterator import TrainIterator


OBS_IDX = 0
ACT_IDX = 1


class EnvParser:

    # parser stages
    NULL = -1
    TRAIN = 0
    EVAL = 2

    # parser states
    THETA_M = 4
    THETA_M2 = 3
    BELIEF = 5
    THETA_T = 6
    STEPS = 7
    OBS = 8
    ACTION = 9

    def __init__(self, filename, params):
        self.params = params
        training_set = self.parse_input_data_file(filename)
        # random.shuffle(training_set)          # disable for consistency between different runs

        num_envs = len(training_set)
        cutoff = int(np.floor(num_envs * params.training_envs))
        self.train_data = training_set[:cutoff]
        self.valid_data = training_set[cutoff:]

    def get_training_iterator(self):
        return TrainIterator(self.params, self.train_data)

    def get_validation_iterator(self):
        return TrainIterator(self.params, self.valid_data)

    @staticmethod
    def parse_input_data_file(filename):
        """
        Read the full input file - to be used during training.
        :param filename:
        :return:
        """
        f = open(filename, "r")

        stage = EnvParser.NULL
        state = EnvParser.NULL

        training_set = None

        theta_m = None
        theta_m2 = None
        theta_t = None
        theta_t_list = None
        b0 = None
        b0_list = None

        trajectory = None
        traj_list = None
        mut_steps = None
        mut_step_list = None

        obs = None

        for line in f:
            line = line.rstrip("\n")
            line = line.rstrip("\r")

            if line[0] == "#":
                # skip header line
                pass

            elif line == "train":
                stage = EnvParser.TRAIN
                state = EnvParser.NULL
                training_set = []

            elif line == "tm":
                if state == EnvParser.ACTION:
                    # if last state was action, then finish environment
                    traj_list.append(trajectory)
                    mut_step_list.append(mut_steps)
                    if theta_m2 is None:
                        theta_m2 = "[]"
                    env_inst = EnvironmentInstance(np.array(eval(theta_m)), np.array(eval(theta_m2)),
                                                   theta_t_list, b0_list, traj_list, mut_step_list)
                    if stage == EnvParser.TRAIN:
                        training_set.append(env_inst)
                    # elif stage == self.EVAL:
                    #     eval_set.append(env_inst)

                # prepare for new environment
                state = EnvParser.THETA_M
                theta_m = ""
                theta_t_list = []
                b0_list = []
                traj_list = []
                mut_step_list = []

            elif line == "tm2":
                # prepare for new theta 2
                theta_m2 = ""
                state = EnvParser.THETA_M2

            elif line == "tt":
                if state == EnvParser.ACTION:
                    # if last state was action, then finish trajectory
                    traj_list.append(trajectory)
                    mut_step_list.append(mut_steps)

                # prepare for new reward map
                state = EnvParser.THETA_T
                theta_t = ""

            elif line == "bel":
                theta_t_list.append(np.array(eval(theta_t)))
                # prepare for new initial belief
                state = EnvParser.BELIEF
                b0 = ""

            elif line == "steps":
                b0_list.append(np.array(eval(b0)))
                # prepare for new trajectory
                state = EnvParser.STEPS
                trajectory = []
                mut_steps = []

            elif line == "mutate":
                # store the current step number as the mutation step number
                mut_idx = len(trajectory)
                mut_steps.append(mut_idx)

            elif line == "end":
                if state == EnvParser.ACTION:
                    # if last state was action, then finish environment
                    if len(traj_list) > 0:
                        traj_list.append(trajectory)
                        mut_step_list.append(mut_steps)
                        if theta_m2 is None:
                            theta_m2 = "[]"
                        env_inst = EnvironmentInstance(np.array(eval(theta_m)), np.array(eval(theta_m2)),
                                                       theta_t_list, b0_list, traj_list, mut_step_list)
                        training_set.append(env_inst)
                break

            else:
                # no keyword - parse based on previous state
                if state == EnvParser.THETA_M:
                    # read line of theta_m matrix
                    theta_m += line
                    # stay in state THETA_M

                elif state == EnvParser.THETA_M2:
                    # read line of theta_m 2 matrix
                    theta_m2 += line
                    # stay in state THETA_M2

                elif state == EnvParser.THETA_T:
                    # read line of theta_t matrix
                    theta_t += line
                    # stay in state THETA_T

                elif state == EnvParser.BELIEF:
                    # read line of belief matrix
                    b0 += line
                    # stay in state BELIEF

                elif state == EnvParser.STEPS:
                    # read observation
                    obs = [int(x) for x in line.split()]
                    # update state
                    state = EnvParser.OBS

                elif state == EnvParser.OBS:
                    # read action
                    a_tgt = int(line)
                    trajectory.append((obs, a_tgt))
                    # update state
                    state = EnvParser.ACTION

                elif state == EnvParser.ACTION:
                    # read observation
                    obs = [int(x) for x in line.split()]
                    # update state
                    state = EnvParser.OBS

        return training_set

    @staticmethod
    def parse_input_header(filename, params):
        """
        Read only the header of the input file - to be used during evaluation.
        :param filename:
        :param params:
        :return:
        """
        f = open(filename, "r")
        for line in f:
            line = line.rstrip("\n")
            line = line.rstrip("\r")
            if line[0] == "#":
                # parse header line
                [key, value] = line.replace("#", "").split(":")
                params[key] = eval(value)
            else:
                break
        f.close()


class EnvironmentInstance:

    def __init__(self, theta_m, theta_m2, theta_t_list, b0_list, trajectories_list, mut_step_list):
        self.theta_m = theta_m
        self.theta_m2 = theta_m2
        self.theta_t_list = theta_t_list
        self.b0_list = b0_list
        self.mut_step_list = mut_step_list
        # trajectory format: [(obs_0, a_tgt_0), (obs_1, a_tgt_1), ... (obs_n, a_tgt_n)]
        self.trajectories_list = trajectories_list

    def is_valid(self):
        if self.theta_m is None:
            return False
        else:
            return True



