"""
Code from https://github.com/sfujim/BCQ
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class ReplayBuffer(Dataset):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6), save_folder=None):
                self.max_size = max_size
                self.ptr = 0
                self.size = 0
                self.max_size = max_size
                self.state_dim = state_dim
                self.action_dim = action_dim
                self.reset()
                self.device = device
                self.save_folder = save_folder

    def reset(self):
                self.state = np.zeros((self.max_size, self.state_dim))
                self.action = np.zeros((self.max_size, self.action_dim))
                self.next_state = np.zeros((self.max_size, self.state_dim))
                self.reward = np.zeros((self.max_size, 1))
                self.not_done = np.zeros((self.max_size, 1))
                self.ptr = 0
                self.size = 0

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, ind=None, seed=0, to_device=True):
                if ind is None: 
                    ind = np.random.randint(0, self.size, size=batch_size)

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


    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.size])
        np.save(f"{save_folder}_action.npy", self.action[:self.size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)


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

        self.state = torch.FloatTensor(self.state).to(self.device)
        self.action = torch.FloatTensor(self.action).to(self.device)
        self.next_state = torch.FloatTensor(self.next_state).to(self.device)
        self.reward = torch.FloatTensor(self.reward).to(self.device)
        self.not_done = torch.FloatTensor(self.not_done).to(self.device)



    def load_indexes(self, indexes):
        reward_buffer = np.load(f"{self.save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(len(indexes)), self.max_size) if len(indexes) > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)

        self.state[:self.size] = np.load(f"{self.save_folder}_state.npy")[indexes]
        self.action[:self.size] = np.load(f"{self.save_folder}_action.npy")[indexes]
        self.next_state[:self.size] = np.load(f"{self.save_folder}_next_state.npy")[indexes]
        self.reward[:self.size] = np.load(f"{self.save_folder}_reward.npy")[indexes]
        self.not_done[:self.size] = np.load(f"{self.save_folder}_not_done.npy")[indexes]


        self.state = torch.FloatTensor(self.state).to(self.device)
        self.action = torch.FloatTensor(self.action).to(self.device)
        self.next_state = torch.FloatTensor(self.next_state).to(self.device)
        self.reward = torch.FloatTensor(self.reward).to(self.device)
        self.not_done = torch.FloatTensor(self.not_done).to(self.device)

        return self.state, self.action, self.next_state, self.reward, self.not_done


    def __getitem__(self, ind):
        # reward_buffer = np.load(f"{self.save_folder}_reward.npy")
        # # Adjust crt_size if we're using a custom size
        # size = min(int(len(indexes)), self.max_size) if len(indexes) > 0 else self.max_size
        # self.size = min(reward_buffer.shape[0], size)

        # state = np.load(f"{self.save_folder}_state.npy")[indexes]
        # action = np.load(f"{self.save_folder}_action.npy")[indexes]
        # next_state = np.load(f"{self.save_folder}_next_state.npy")[indexes]
        # reward = np.load(f"{self.save_folder}_reward.npy")[indexes]
        # not_done = np.load(f"{self.save_folder}_not_done.npy")[indexes]

        # return (state, action, next_state, reward, not_done)

        return self.state[ind], self.action[ind], self.next_state[ind], self.reward[ind], self.not_done[ind]


    def get_all(self):
        return (self.state, self.action, self.next_state, self.reward, self.not_done)




    def __len__(self):
        return self.size
