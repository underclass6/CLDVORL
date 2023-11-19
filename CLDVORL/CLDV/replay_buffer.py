"""
Code from https://github.com/sfujim/BCQ
"""
import numpy as np
from numpy.random import default_rng
import torch
from pathlib import Path

class ReplayBuffer():
    def __init__(self, state_dim, action_dim, device, max_size=int(5e6)):
                self.max_size = max_size
                self.ptr = 0
                self.size = 0
                self.current = 0
                self.max_size = max_size
                self.state_dim = state_dim
                self.action_dim = action_dim
                self.indices = None

                self.action_mean = None
                self.action_std = None
                self.state_mean = None
                self.state_std = None

                self.trajs = []
                self.trajs_size = 0

                self.reset()

                self.device = device

    def reset(self):
                self.state = np.zeros((self.max_size, self.state_dim))
                self.action = np.zeros((self.max_size, self.action_dim))
                self.next_state = np.zeros((self.max_size, self.state_dim))
                self.reward = np.zeros((self.max_size, 1))
                self.not_done = np.zeros((self.max_size, 1))
                self.ptr = 0
                self.size = 0
                self.current = 0
    def reset_index(self):
        self.current = 0

    def get_indices(self):
        return self.indices

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def shuffle(self):
        indices = np.random.permutation(self.size)
        self.indices = indices
        self.state = self.state[indices]
        self.action = self.action[indices]
        self.next_state = self.next_state[indices]
        self.reward = self.reward[indices]
        self.not_done = self.not_done[indices]

    def sample(self, batch_size, ind=None, random_s=False, to_device=True):
        if random_s:
            ind = np.random.randint(0, self.size, size=batch_size)

        if (ind is None) and not random_s:
            max_index = self.current + batch_size
            ind = np.array([i if i < self.size else i - self.size
                       for i in range(self.current, max_index)])
            self.current = max_index % self.size
            #
            # if max_index >= self.size - 1:
            #     self.reset_index()
                # self.shuffle()


        if to_device:
            return (torch.FloatTensor(self.state[ind]).to(self.device),
                    torch.FloatTensor(self.action[ind]).to(self.device),
                    torch.FloatTensor(self.next_state[ind]).to(self.device),
                    torch.FloatTensor(self.reward[ind]).to(self.device),
                    torch.FloatTensor(self.not_done[ind]).to(self.device))
        else:
            return (self.state[ind],
                    self.action[ind],
                    self.next_state[ind],
                    self.reward[ind],
                    self.not_done[ind])


    def sample_trajectory(self, ind=None, to_device=True):
        if ind is None:
            ind = np.random.randint(0, self.trajs_size, size=1)
            ind = ind[0]

        traj_dict = self.trajs[ind]
        state, action, next_state, reward, not_done = traj_dict["state"], traj_dict["action"], traj_dict["next_state"], traj_dict["reward"], traj_dict["not_done"]


        if to_device:
            return (torch.FloatTensor(state).to(self.device),
                    torch.FloatTensor(action).to(self.device),
                    torch.FloatTensor(next_state).to(self.device),
                    torch.FloatTensor(reward).to(self.device),
                    torch.FloatTensor(not_done).to(self.device))
        else:
            return (state,
                    action,
                    next_state,
                    reward,
                    not_done)

    def sample_n_trajectories(self, traj_num=100, ind=None, total_length=200, to_device=True):
        # if random_s:
        ind = np.random.randint(0, self.trajs_size, size=traj_num)

        # trajs_len = [int(self.trajs[idx]["length"]) for idx in ind]
        # print(trajs_len)
        # print(sum(trajs_len))

        # current_length = 0
        # sel_ind = []
        # i = 0
        # while current_length < total_length:
        #     sel_ind.append(i)
        #     current_length+=trajs_len[i]
        #     i+=1



        state, action, next_state, reward, not_done = self.concat_trajectories([self.trajs[idx] for idx in ind])

        if to_device:
            return (torch.FloatTensor(state).to(self.device),
                    torch.FloatTensor(action).to(self.device),
                    torch.FloatTensor(next_state).to(self.device),
                    torch.FloatTensor(reward).to(self.device),
                    torch.FloatTensor(not_done).to(self.device))
        else:
            return (state,
                    action,
                    next_state,
                    reward,
                    not_done)

    def concat_trajectories(self, trajs: list):
        """
            Take a list of trajectory dictionaries and return separate arrays
        """
        state = np.concatenate([traj["state"] for traj in trajs])
        action = np.concatenate([traj["action"] for traj in trajs])
        next_state = np.concatenate([traj["next_state"] for traj in trajs])
        reward = np.concatenate([traj["reward"] for traj in trajs])
        not_done = np.concatenate([traj["not_done"] for traj in trajs])

        return state, action, next_state, reward, not_done


    def save(self, save_folder):
        new_source_path  = save_folder+"/new_source_buffer/"
        Path(new_source_path).mkdir(parents=True, exist_ok=True)
        np.save(f"{new_source_path}_state.npy", self.state[:self.size])
        np.save(f"{new_source_path}_action.npy", self.action[:self.size])
        np.save(f"{new_source_path}_next_state.npy", self.next_state[:self.size])
        np.save(f"{new_source_path}_reward.npy", self.reward[:self.size])
        np.save(f"{new_source_path}_not_done.npy", self.not_done[:self.size])
        np.save(f"{new_source_path}_ptr.npy", self.ptr)


    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)

        self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
        self.reward[:self.size] = reward_buffer[:self.size]
        self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]

        self.action_mean =  torch.FloatTensor(np.mean(self.action, axis=0)).to(self.device)
        self.action_std = torch.FloatTensor(np.std(self.action, axis=0)).to(self.device)

        self.create_trajs()

    def create_trajs(self):
        terminal_states = np.where(self.not_done[:self.size] == 0.)[0]
        terminal_states = np.append(terminal_states, self.size-1)

        min_ind = 0
        for max_ind in terminal_states:
            state = self.state[min_ind:max_ind + 1]
            action = self.action[min_ind:max_ind + 1]
            next_state = self.next_state[min_ind:max_ind + 1]
            reward = self.reward[min_ind:max_ind + 1]
            not_done = self.not_done[min_ind:max_ind + 1]

            # self.trajs.append([state, action, next_state, reward, not_done])
            self.trajs.append({"state": state,
                              "action": action,
                              "next_state": next_state,
                              "reward": reward,
                              "not_done": not_done,
                              "length": max_ind - min_ind + 1})

            min_ind = max_ind + 1

        self.trajs_size = len(self.trajs)


    def load_mix(self, source_folder, target_folder, size=200000):
        reward_buffer = np.load(f"{source_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        # size = min(int(size), self.max_size) if size > 0 else self.max_size
        # self.size = min(reward_buffer.shape[0], size)

        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)
        s_size = int(size / 2)

        self.state[:s_size] = np.load(f"{source_folder}_state.npy")[:s_size]
        self.action[:s_size] = np.load(f"{source_folder}_action.npy")[:s_size]
        self.next_state[:s_size] = np.load(f"{source_folder}_next_state.npy")[:s_size]
        self.reward[:s_size] = np.load(f"{source_folder}_reward.npy")[:s_size]
        self.not_done[:s_size] = np.load(f"{source_folder}_not_done.npy")[:s_size]

        self.state[s_size:] = np.load(f"{target_folder}_state.npy")[:s_size]
        self.action[s_size:] = np.load(f"{target_folder}_action.npy")[:s_size]
        self.next_state[s_size:] = np.load(f"{target_folder}_next_state.npy")[:s_size]
        self.reward[s_size:] = np.load(f"{target_folder}_reward.npy")[:s_size]
        self.not_done[s_size:] = np.load(f"{target_folder}_not_done.npy")[:s_size]


    def load_indexes(self, save_folder, indexes):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(len(indexes)), self.max_size) if len(indexes) > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)

        self.state[:self.size] = np.load(f"{save_folder}_state.npy")[indexes]
        self.action[:self.size] = np.load(f"{save_folder}_action.npy")[indexes]
        self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[indexes]
        self.reward[:self.size] = np.load(f"{save_folder}_reward.npy")[indexes]
        self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[indexes]

    def merge_buffers(self, buff1, buff2):
        self.state = np.concatenate((buff1.state, buff2.state), axis=0)
        self.action = np.concatenate((buff1.action, buff2.action), axis=0)
        self.next_state = np.concatenate((buff1.next_state, buff2.next_state), axis=0)
        self.reward = np.concatenate((buff1.reward, buff2.reward), axis=0)
        self.not_done = np.concatenate((buff1.not_done, buff2.not_done), axis=0)

        self.size = buff1.size + buff2.size

    def __len__(self):
        return self.size