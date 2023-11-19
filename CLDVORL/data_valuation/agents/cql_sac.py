import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Normal
import numpy as np
import math
import copy
from tqdm import tqdm



def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob

    def select_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        return action.detach().cpu()

    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CQLSAC(object):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                        state_size,
                        action_size,
                        tau,
                        hidden_size,
                        learning_rate,
                        temp,
                        with_lagrange,
                        cql_weight,
                        target_action_gap,
                        device
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(CQLSAC, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.device = device
        
        self.gamma = torch.FloatTensor([0.99]).to(device)
        self.tau = tau
        hidden_size = hidden_size
        learning_rate = learning_rate
        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 
        
        # CQL params
        self.with_lagrange = with_lagrange
        self.temp = temp
        self.cql_weight = cql_weight
        self.target_action_gap = target_action_gap
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate) 
        
        # Actor Network 

        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = Critic(state_size, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(state_size, action_size, hidden_size, 1).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 

    
    def select_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            if eval:
                action = self.actor_local.get_det_action(state)
            else:
                action = self.actor_local.select_action(state)
        return action.numpy()

    # def select_actions(self, state, epsilon = 1e-6):
    #     # state = torch.from_numpy(state).float().to(self.device)
    #     state = torch.FloatTensor(state).to(self.device)
    #
    #     mu, log_std = self.actor_local.forward(state)
    #     std = log_std.exp()
    #     dist = Normal(mu, std)
    #     e = dist.rsample().to(state.device)
    #     action = torch.tanh(e)
    #     return action.cpu()
    def select_actions(self, state, eval=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            if eval:
                action = self.actor_local.get_det_action(state)
            else:
                action = self.actor_local.select_action(state)
        return action.cpu()

    def calc_policy_loss(self, states, alpha):
        actions_pred, log_pis = self.actor_local.evaluate(states)

        # q1 = self.critic1(states, actions_pred.squeeze(0))
        # q2 = self.critic2(states, actions_pred.squeeze(0))
        q1 = self.critic1(states, actions_pred)
        q2 = self.critic2(states, actions_pred)
        min_Q = torch.min(q1,q2).cpu()
        actor_loss = ((alpha * log_pis.cpu() - min_Q )).mean()
        return actor_loss, log_pis

    def _compute_policy_values(self, obs_pi, obs_q):
        #with torch.no_grad():
        actions_pred, log_pis = self.actor_local.evaluate(obs_pi)
        
        qs1 = self.critic1(obs_q, actions_pred)
        qs2 = self.critic2(obs_q, actions_pred)
        
        return qs1 - log_pis.detach(), qs2 - log_pis.detach()
    
    def _compute_random_values(self, obs, actions, critic):
        random_values = critic(obs, actions)
        random_log_probs = math.log(0.5 ** self.action_size)
        return random_values - random_log_probs
    
    def train(self,  replay_buffer, iterations, batch_size=256, random_s=False, disable_tqdm=True):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # states, actions, rewards, next_states, dones = experiences

        for it in tqdm(range(iterations), disable=disable_tqdm):
            states, actions, next_states, rewards, not_dones = replay_buffer.sample(batch_size, random_s=random_s, to_device=True)

            # ---------------------------- update actor ---------------------------- #
            current_alpha = copy.deepcopy(self.alpha)
            actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Compute alpha loss
            alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            with torch.no_grad():
                next_action, new_log_pi = self.actor_local.evaluate(next_states)
                Q_target1_next = self.critic1_target(next_states, next_action)
                Q_target2_next = self.critic2_target(next_states, next_action)
                Q_target_next = torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * new_log_pi
                # Compute Q targets for current states (y_i)
                Q_targets = rewards + (self.gamma * not_dones * Q_target_next)


            # Compute critic loss
            q1 = self.critic1(states, actions)
            q2 = self.critic2(states, actions)

            critic1_loss = F.mse_loss(q1, Q_targets)
            critic2_loss = F.mse_loss(q2, Q_targets)

            # CQL addon
            random_actions = torch.FloatTensor(q1.shape[0] * 10, actions.shape[-1]).uniform_(-1, 1).to(self.device)
            num_repeat = int (random_actions.shape[0] / states.shape[0])
            temp_states = states.unsqueeze(1).repeat(1, num_repeat, 1).view(states.shape[0] * num_repeat, states.shape[1])
            temp_next_states = next_states.unsqueeze(1).repeat(1, num_repeat, 1).view(next_states.shape[0] * num_repeat, next_states.shape[1])

            current_pi_values1, current_pi_values2  = self._compute_policy_values(temp_states, temp_states)
            next_pi_values1, next_pi_values2 = self._compute_policy_values(temp_next_states, temp_states)

            random_values1 = self._compute_random_values(temp_states, random_actions, self.critic1).reshape(states.shape[0], num_repeat, 1)
            random_values2 = self._compute_random_values(temp_states, random_actions, self.critic2).reshape(states.shape[0], num_repeat, 1)

            current_pi_values1 = current_pi_values1.reshape(states.shape[0], num_repeat, 1)
            current_pi_values2 = current_pi_values2.reshape(states.shape[0], num_repeat, 1)

            next_pi_values1 = next_pi_values1.reshape(states.shape[0], num_repeat, 1)
            next_pi_values2 = next_pi_values2.reshape(states.shape[0], num_repeat, 1)

            cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], 1)
            cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], 1)

            assert cat_q1.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q1 instead has shape: {cat_q1.shape}"
            assert cat_q2.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q2 instead has shape: {cat_q2.shape}"


            cql1_scaled_loss = ((torch.logsumexp(cat_q1 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - q1.mean()) * self.cql_weight
            cql2_scaled_loss = ((torch.logsumexp(cat_q2 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - q2.mean()) * self.cql_weight

            cql_alpha_loss = torch.FloatTensor([0.0])
            cql_alpha = torch.FloatTensor([0.0])
            if self.with_lagrange:
                cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
                cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
                cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

                self.cql_alpha_optimizer.zero_grad()
                cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5
                cql_alpha_loss.backward(retain_graph=True)
                self.cql_alpha_optimizer.step()

            total_c1_loss = critic1_loss + cql1_scaled_loss
            total_c2_loss = critic2_loss + cql2_scaled_loss


            # Update critics
            # critic 1
            self.critic1_optimizer.zero_grad()
            total_c1_loss.backward(retain_graph=True)
            clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
            self.critic1_optimizer.step()
            # critic 2
            self.critic2_optimizer.zero_grad()
            total_c2_loss.backward()
            clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
            self.critic2_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)
        
        # return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(), cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item()

    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


    def save(self, filename, type):
        torch.save(self.critic1.state_dict(), filename + f"_cql_critic1_{type}")
        torch.save(self.critic1_optimizer.state_dict(), filename + f"_cql_critic1_optimizer_{type}")

        torch.save(self.critic2.state_dict(), filename + f"_cql_critic2_{type}")
        torch.save(self.critic2_optimizer.state_dict(), filename + f"_cql_critic2_optimizer_{type}")

        torch.save(self.actor_local.state_dict(), filename + f"_cql_actor_{type}")
        torch.save(self.actor_optimizer.state_dict(), filename + f"_cql_actor_optimizer_{type}")



    def load(self, filename, type):
        self.critic1.load_state_dict(torch.load(filename + f"_cql_critic1_{type}", map_location=torch.device(self.device)))
        self.critic1_optimizer.load_state_dict(torch.load(filename + f"_cql_critic1_optimizer_{type}", map_location=torch.device(self.device)))
        # self.critic1_target = copy.deepcopy(self.critic1)

        self.critic2.load_state_dict(torch.load(filename + f"_cql_critic2_{type}", map_location=torch.device(self.device)))
        self.critic2_optimizer.load_state_dict(torch.load(filename + f"_cql_critic2_optimizer_{type}", map_location=torch.device(self.device)))
        # self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_local.load_state_dict(torch.load(filename + f"_cql_actor_{type}", map_location=torch.device(self.device)))
        self.actor_optimizer.load_state_dict(torch.load(filename + f"_cql_actor_optimizer_{type}", map_location=torch.device(self.device)))
        # self.actor_target = copy.deepcopy(self.actor_local)
