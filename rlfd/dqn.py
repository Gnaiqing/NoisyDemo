"""
The code is adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
The code assumes a 1D box observation space and 1D discrete action space
"""
import gym
from gym.spaces.utils import flatdim, flatten_space
import math
import random
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from pathlib import Path
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

BATCH_SIZE = 32
POLICY_UPDATE = 4
GAMMA = 0.95
EPS_START = 1
EPS_END = 0.01
TARGET_UPDATE = 100

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, dim_in, dim_out, hidden_layer_sizes=(64,), activation="tanh"):
        super(DQN, self).__init__()
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError("Activation function not implemented.")

        self.layers = []
        self.layers.append(nn.Linear(dim_in, hidden_layer_sizes[0]))
        self.layers.append(self.act)
        for i in range(len(hidden_layer_sizes) - 1):
            self.layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            self.layers.append(self.act)

        self.layers.append(nn.Linear(hidden_layer_sizes[-1], dim_out))

        self.net = nn.Sequential(
            *self.layers
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        y = self.net(x.float())
        return y


def select_action(policy_net, state, n_actions, epsilon):
    sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #                 math.exp(-1. * steps_done / EPS_DECAY)
    eps_threshold = epsilon
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).argmax().view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(episode_durations, figpath):
    plt.figure()
    plt.plot(episode_durations)
    plt.title('Episode durations')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.savefig(figpath)


def plot_rewards(episode_rewards, figpath):
    plt.figure()
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Rewards')
    plt.savefig(figpath)


def plot_loss(losses, figpath):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Losses")
    plt.savefig(figpath)


def plot_scores(scores, figpath, num_noisy):
    plt.figure()
    scores = np.stack(scores, 0)
    n_demos = scores.shape[1]
    for i in range(n_demos):
        plt.plot(scores[:, i], 'g' if i < num_noisy else 'b', label=f"demo-{i}")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.savefig(figpath)


def optimize_model(policy_net, target_net, optimizer, memory):
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    n_dim = len(batch.state[0])
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                         batch.next_state)), device=device, dtype=torch.bool)
    # non_final_next_states = torch.cat([s for s in batch.next_state
    #                                    if s is not None]).reshape(-1, n_dim)
    state_batch = torch.cat(batch.state).reshape(-1, n_dim)
    next_state_batch = torch.cat(batch.next_state).reshape(-1, n_dim)
    action_batch = torch.cat(batch.action).reshape(-1,1)
    reward_batch = torch.cat(batch.reward)
    done_batch = torch.cat(batch.done)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = target_net(next_state_batch).max(dim=1)[0] * (1-done_batch).detach()
    # Compute the expected Q values
    expected_state_action_values = ((next_state_values * GAMMA) + reward_batch).unsqueeze(1)
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)
    loss_value = loss.detach().to("cpu").numpy()

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss_value.item()


def train_dqn(env, num_episodes, eps_decay):
    n_actions = env.action_space.n
    n_dim = flatdim(env.observation_space)
    policy_net = DQN(n_dim, n_actions).to(device)
    target_net = DQN(n_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    # optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)
    losses = []
    steps_to_update_model = 0
    # first fill the replay buffer
    index = 0
    exp_replay_size = 10000
    for t in range(exp_replay_size):
        state = env.reset()
        state = torch.tensor(state, device=device)
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            action = torch.tensor([action], device=device)
            reward = torch.tensor([reward], device=device)
            dones = torch.tensor([int(done)], device=device)
            next_state = torch.tensor(next_state, device=device)
            memory.push(state, action, next_state, reward, dones)
            state = next_state
            index += 1
            if index > exp_replay_size:
                break

    epsilon = EPS_START
    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and state
        state = env.reset()
        state = torch.tensor(state, device=device)
        for t in count():
            # Select and perform an action
            action = select_action(policy_net, state, n_actions, epsilon)
            epsilon = max(epsilon * eps_decay, EPS_END)
            next_state, reward, done, _ = env.step(action.item())
            action = torch.tensor([action], device=device)
            reward = torch.tensor([reward], device=device)
            dones = torch.tensor([int(done)], device=device)
            #TODO: add done info in computing rewards
            next_state = torch.tensor(next_state, device=device)

            # Store the transition in memory
            memory.push(state, action, next_state, reward, dones)

            # Move to the next state
            state = next_state
            steps_to_update_model += 1
            # Perform one step of the optimization (on the policy network)
            if steps_to_update_model % POLICY_UPDATE == 0:
                loss = optimize_model(policy_net, target_net, optimizer, memory)
                losses.append(loss)
            # Update the target network, copying all weights and biases in DQN
            if steps_to_update_model >= TARGET_UPDATE:
                target_net.load_state_dict(policy_net.state_dict())
                steps_to_update_model = 0

            if done:
                episode_durations.append(t + 1)
                break

    print('Complete')
    return policy_net, episode_durations, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--n_episodes", type=int, default=1000)
    parser.add_argument("--eps_decay", type=float, default=0.95)
    parser.add_argument("--figpath", type=str, default="../fig")
    args = parser.parse_args()
    env = gym.make(args.env)
    policy_net, episode_durations, losses = train_dqn(env, args.n_episodes, args.eps_decay)
    env.close()
    figpath = Path(args.figpath) / f"{args.env}_dqn_duration.jpg"
    plot_durations(episode_durations, figpath)
    figpath_2 = Path(args.figpath) / f"{args.env}_dqn_loss.jpg"
    plot_loss(losses, figpath_2)


