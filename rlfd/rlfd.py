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
from dqn import DQN, ReplayMemory, select_action, plot_durations, plot_loss
from potential_function import load_demonstrations, calc_potential

BATCH_SIZE = 32
POLICY_UPDATE = 4
GAMMA = 0.95
EPS_START = 1
EPS_END = 0.01
TARGET_UPDATE = 100
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))
episode_durations = []


def optimize_rlfd_model(policy_net, target_net, optimizer, memory, demo_dict, low, high, sigma=0.5):
    """
    Optimize the model using demonstration and reward
    :param policy_net:
    :param target_net:
    :param optimizer:
    :param memory:
    :param demo_dict:
    :param sigma:
    :return:
    """
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    n_dim = len(batch.state[0])
    # Compute a mask of non-final states and concatenate the batch elements
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
    next_action_batch = policy_net(next_state_batch).max(dim=1)[1]
    batch_f = []
    for i in range(len(batch.state)):
        s = state_batch[i].detach().numpy()
        a = action_batch[i].detach().numpy().item()
        next_s = next_state_batch[i].detach().numpy()
        next_a = next_action_batch[i].detach().numpy().item()
        f = GAMMA * calc_potential(next_s,next_a,low, high, demo_dict, sigma=sigma) - \
            calc_potential(s,a,low, high, demo_dict, sigma=sigma)
        batch_f.append(f)

    batch_f = torch.tensor(batch_f).to(device)
    next_state_values = (target_net(next_state_batch).max(dim=1)[0] * (1-done_batch)).detach()
    # Compute the expected Q values
    expected_state_action_values = ((next_state_values * GAMMA) + reward_batch + batch_f).unsqueeze(1)
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


def train_rlfd(env, demo_dict, num_episodes):
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
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = env.reset()
        state = torch.tensor(state, device=device)
        for t in count():
            # Select and perform an action
            action = select_action(policy_net, state, n_actions, epsilon)
            epsilon = max(epsilon * 0.999, EPS_END)
            next_state, reward, done, _ = env.step(action.item())
            action = torch.tensor([action], device=device)
            reward = torch.tensor([reward], device=device)
            dones = torch.tensor([int(done)], device=device)

            next_state = torch.tensor(next_state, device=device)

            # Store the transition in memory
            memory.push(state, action, next_state, reward, dones)

            # Move to the next state
            state = next_state
            steps_to_update_model += 1
            # Perform one step of the optimization (on the policy network)
            if steps_to_update_model % POLICY_UPDATE == 0:
                loss = optimize_rlfd_model(policy_net, target_net, optimizer, memory, demo_dict,
                                           env.observation_space.low, env.observation_space.high)
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
    parser.add_argument("--dataset_dir", type=str, default="../datasets")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--expert_model", type=str, default="A2C")
    parser.add_argument("--n_experts", type=int, default=10)
    parser.add_argument("--n_episodes", type=int, default=1000)
    parser.add_argument("--figpath", type=str, default="../fig")
    args = parser.parse_args()
    env = gym.make(args.env)
    demo_dict = load_demonstrations(args.dataset_dir, args.expert_model, args.env, args.n_experts)
    policy_net, episode_durations, losses = train_rlfd(env, demo_dict, args.n_episodes)
    env.close()
    figpath = Path(args.figpath) / f"{args.env}_rlfd_duration.jpg"
    plot_durations(episode_durations, figpath)
    figpath_2 = Path(args.figpath) / f"{args.env}_rlfd_loss.jpg"
    plot_loss(losses, figpath_2)