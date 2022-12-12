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
from dqn import DQN, ReplayMemory, select_action, plot_durations, plot_loss, plot_scores, plot_rewards
from potential_function import load_demonstrations, calc_potential, calc_potential_scored
from tqdm import tqdm

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



def optimize_rlfd_model(ind, policy_net, target_net, optimizer, memory, demos_states, demos_actions, demo_dict,
                        last_scores, n_iter, low, high, sigma=0.5):
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

    n_demos = len(demo_dict)
    scores = []
    for i in range(n_demos):
        # print(demos_states[i].device)
        # print(demos_states[i].shape, demos_actions[i].shape, policy_net(demos_states[i]).shape)
        score = (policy_net(demos_states[i]).gather(1, demos_actions[i].reshape(-1, 1)) - target_net(demos_states[i])).mean().detach().cpu().item()
        scores.append(score)
    # print(scores)
    scores = np.exp(100 * np.array(scores))
    scores = scores / np.sum(scores)
    scores = last_scores * n_iter + scores
    scores /= scores.sum()
    # print(scores)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_action_batch = policy_net(next_state_batch).max(dim=1)[1]
    batch_f = []
    for i in range(len(batch.state)):
        # s = state_batch[i].detach().cpu().numpy()
        # a = action_batch[i].detach().cpu().numpy().item()
        # next_s = next_state_batch[i].detach().cpu().numpy()
        # next_a = next_action_batch[i].detach().cpu().numpy().item()
        s = state_batch[i].detach()
        a = action_batch[i].detach()
        next_s = next_state_batch[i].detach()
        next_a = next_action_batch[i].detach()
        if ind:
            f = GAMMA * calc_potential_scored(next_s,next_a.detach().cpu().item(),low, high, demo_dict, scores, sigma=sigma) - \
                calc_potential_scored(s,a.detach().cpu().item(),low, high, demo_dict, scores, sigma=sigma)
        else:
            f = GAMMA * calc_potential(next_s,next_a.detach().cpu().item(),low, high, demo_dict, sigma=sigma) - \
                calc_potential(s,a.detach().cpu().item(),low, high, demo_dict, sigma=sigma)
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
    return loss_value.item(), scores


def train_rlfd(ind, env, demos_states, demos_actions, demo_dict, num_episodes, eps_decay):
    # pbar = InitBar()

    n_actions = env.action_space.n
    n_dim = flatdim(env.observation_space)

    n_demos = len(demo_dict)
    for i in range(n_demos):
        demos_states[i] = torch.tensor(demos_states[i], dtype=torch.float).to(device)
        demos_actions[i] = torch.tensor(demos_actions[i], dtype=torch.int64).to(device)
        for action in demo_dict[i].keys():
            demo_dict[i][action] = torch.tensor(demo_dict[i][action], dtype=torch.float).to(device)

    ob_low = torch.tensor(env.observation_space.low, dtype=torch.float).to(device)
    ob_high = torch.tensor(env.observation_space.high, dtype=torch.float).to(device)

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
    last_scores = np.zeros(n_demos)
    scores = []
    episode_durations = []
    episode_rewards = []
    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and state
        state = env.reset()
        state = torch.tensor(state, device=device)
        sum_reward = 0
        for t in count():
            # Select and perform an action
            action = select_action(policy_net, state, n_actions, epsilon)
            epsilon = max(epsilon * eps_decay, EPS_END)
            next_state, reward, done, _ = env.step(action.item())
            sum_reward += reward
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
                loss, last_scores = optimize_rlfd_model(ind, policy_net, target_net, optimizer, memory, demos_states, demos_actions, demo_dict, last_scores, i_episode,
                                           ob_low, ob_high)
                scores.append(last_scores)
                losses.append(loss)

            # Update the target network, copying all weights and biases in DQN
            if steps_to_update_model >= TARGET_UPDATE:
                target_net.load_state_dict(policy_net.state_dict())
                steps_to_update_model = 0

            if done:
                episode_durations.append(t + 1)
                episode_rewards.append(sum_reward)
                break

        # pbar((i_episode + 1) / num_episodes * 100)

    print('Complete')
    return policy_net, episode_durations, episode_rewards, losses, scores


def evaluate_rlfd(policy_net, env, criterion="reward", repeats=100):
    """
    Evaluate the performance of trained policy
    :param policy_net: trained network
    :param env: environment
    :param criterion: "episode" or "reward"
    :return:
    """
    rewards_list = []
    durations_list = []
    for i in range(repeats):
        state = env.reset()
        state = torch.tensor(state, device=device)
        done = False
        sum_reward = 0
        duration = 0
        while not done:
            action = policy_net(state).argmax().detach().numpy().item()
            next_state, reward, done, _ = env.step(action)
            state = torch.tensor(state, device=device)
            duration += 1
            sum_reward += reward

        durations_list.append(duration)
        rewards_list.append(sum_reward)

    if criterion == "reward":
        return np.array(rewards_list).mean()
    else:
        return np.array(durations_list).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../datasets")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--expert_model", type=str, default="A2C")
    parser.add_argument("--n_experts", type=int, default=10)
    parser.add_argument("--n_episodes", type=int, default=1000)
    parser.add_argument("--eps_decay", type=float, default=0.95)
    parser.add_argument("--toggle_new_method", type=int, default=1)
    parser.add_argument("--figpath", type=str, default="../fig")
    parser.add_argument("--num_noisy", type=int, default=5)
    args = parser.parse_args()
    print(args.toggle_new_method)
    env = gym.make(args.env)
    demos_states, demos_actions, demo_dict = load_demonstrations(args.dataset_dir, args.expert_model, args.env, args.n_experts)
    policy_net, episode_durations, episode_rewards, losses, scores = train_rlfd(args.toggle_new_method, env, demos_states, demos_actions, demo_dict, args.n_episodes, args.eps_decay)
    env.close()
    figpath = Path(args.figpath) / f"{args.env}_rlfd{'_new' if args.toggle_new_method else ''}_duration.jpg"
    plot_durations(episode_durations, figpath)
    figpath_2 = Path(args.figpath) / f"{args.env}_rlfd{'_new' if args.toggle_new_method else ''}_loss.jpg"
    plot_loss(losses, figpath_2)
    figpath_3 = Path(args.figpath) / f"{args.env}_rlfd{'_new' if args.toggle_new_method else ''}_scores.jpg"
    plot_scores(scores, figpath_3, args.num_noisy)
    figpath_4 = Path(args.figpath) / f"{args.env}_rlfd{'_new' if args.toggle_new_method else ''}_reward.jpg"
    plot_rewards(episode_rewards, figpath_4)

    test_reward = evaluate_rlfd(policy_net, env)
    print("Test reward:", test_reward)
