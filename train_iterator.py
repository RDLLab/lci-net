import numpy as np

OBS_IDX = 0
ACT_IDX = 1


class TrainIterator:
    # no difference to v1, but now separated from parser and eval iterator

    def __init__(self, params, data):
        self.params = params
        self.env_idx = np.zeros([params.batch_size], dtype=np.int8)
        self.traj_idx = np.zeros([params.batch_size], dtype=np.int8)
        self.step_idx = np.zeros([params.batch_size], dtype=np.int8)

        # track which streams are mutated
        self.mutated = np.zeros([params.batch_size], dtype=np.int8)

        # organise data set into batches of size batch_size
        self.num_batches = int(np.floor((1.0 * len(data)) / params.batch_size))

        # discard extra environments (number of envs generated should be divisible by batch_size)
        data = data[:self.num_batches * params.batch_size]
        self.data = np.reshape(data, [params.batch_size, self.num_batches])

        self.done = np.zeros([params.batch_size], dtype=np.int8)
        self.streams_complete = 0

    def next(self):
        """
        Iterate over data set.
        :return: next element of data set
        """
        map = []
        goal = []
        b0 = []
        isstart = []
        act_in = []
        obs_in = []
        weight = []
        act_label = []

        for b in range(self.params.batch_size):
            if not self.done[b]:
                current_env = self.data[b][self.env_idx[b]]
                current_traj = current_env.trajectories_list[self.traj_idx[b]]
                current_mut_steps = current_env.mut_step_list[self.traj_idx[b]]
            else:
                # use first env/traj if done - weight will be zero
                current_env = self.data[b][0]
                current_traj = current_env.trajectories_list[0]
                current_mut_steps = current_env.mut_step_list[0]

            # compute data for this element of the batch
            if self.mutated[b]:
                map.append(current_env.theta_m2)
            else:
                map.append(current_env.theta_m)
            goal.append(current_env.theta_t_list[self.traj_idx[b]])
            b0.append(current_env.b0_list[self.traj_idx[b]])
            if self.step_idx[b] == 0:
                isstart.append(1)
            else:
                isstart.append(0)

            current_act_in = []
            current_obs_in = []
            current_weight = []
            current_act_label = []
            next_traj = False   # flag to move to next trajectory
            for idx in range(self.step_idx[b], self.step_idx[b] + self.params.step_size):
                if self.done[b] == 0 and idx < min(len(current_traj), self.params.lim_traj_len):
                    # not past the last step of the trajectory
                    current_weight.append(1)

                    # add act_in for this step
                    if idx > 0:
                        # act_in is previous expert action
                        current_act_in.append(current_traj[idx - 1][ACT_IDX])
                    else:
                        # if first step, a_in is STAY
                        current_act_in.append(self.params.stay_action)

                    # add obs_in for this step
                    current_obs_in.append(current_traj[idx][OBS_IDX])

                    # add expert action for this step
                    current_act_label.append(current_traj[idx][ACT_IDX])
                else:
                    # past the last step - do not count
                    current_weight.append(0)
                    current_act_in.append(0)
                    current_obs_in.append(current_traj[-1][OBS_IDX])
                    current_act_label.append(0)

                    # move to next trajectory on the next call
                    next_traj = True

            self.step_idx[b] += 4
            if self.step_idx[b] >= len(current_traj):
                next_traj = True

            # store act_in, obs_in, weight, act_label for this set of steps
            act_in.append(current_act_in)
            obs_in.append(current_obs_in)
            weight.append(current_weight)
            act_label.append(current_act_label)

            # mutate if required
            if self.step_idx[b] in current_mut_steps:
                self.mutated[b] = (self.mutated[b] + 1) % 2

            # move to next trajectory if necessary
            if next_traj and not self.done[b]:
                self.traj_idx[b] += 1
                self.step_idx[b] = 0
                self.mutated[b] = 0

                # if at last trajectory, go to next environment
                if self.traj_idx[b] >= len(current_env.trajectories_list):
                    self.traj_idx[b] = 0
                    self.env_idx[b] += 1

                    # if at last environment, signal done and return 0 weight from this point on
                    if self.env_idx[b] >= self.num_batches:
                        # stop here
                        self.done[b] = 1
                        self.streams_complete += 1

        # swap step index and batch index axes
        act_in = np.swapaxes(act_in, 0, 1)
        obs_in = np.swapaxes(obs_in, 0, 1)
        weight = np.swapaxes(weight, 0, 1)
        act_label = np.swapaxes(act_label, 0, 1)

        return [map, goal, b0, isstart, act_in, obs_in, weight, act_label]

    def reset(self):
        """
        Reset iterator to start of data set
        """
        # reset all indices
        self.env_idx = np.zeros([self.params.batch_size], dtype=np.int8)
        self.traj_idx = np.zeros([self.params.batch_size], dtype=np.int8)
        self.step_idx = np.zeros([self.params.batch_size], dtype=np.int8)
        self.done = np.zeros([self.params.batch_size], dtype=np.int8)
        self.streams_complete = 0

    def is_finished(self):
        """
        True if all data has been output.
        :return: True if finished
        """
        return self.streams_complete == self.params.batch_size






