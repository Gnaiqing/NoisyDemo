import gym
import json
import numpy as np
import torch
from pathlib import Path


def normalize_box_state(obs, low, high):
    """
    Normalize observation to range [0,1]
    :param obs: observation or list of observations of shape (n_states, n_dim)
    :param low: lowest bound of space
    :param high: highest bound of space
    :return: x: normalized value in [0,1]
    """
    # obs = np.array(obs)
    x = (obs - low) / (high - low)
    return x


def calc_distance(x1, x2, metric="l2"):
    if metric == "l2":
        return torch.sum(torch.square(x2-x1), dim=1)
    else:
        raise NotImplementedError("Distance metric not implemented.")


def load_demonstrations(dataset_dir, model, env, n_expert):
    """
    Load simulated expert demonstrations
    :param dataset_dir:
    :param model:
    :param env:
    :param n_expert:
    :return:
    """
    demos = []
    for i in range(n_expert):
        filepath = Path(dataset_dir) / f"{model}_{env}_{i}.json"
        infile = open(filepath)
        demo = json.load(infile)
        demos.append(demo)
    return convert_demos_to_dict(demos)


def convert_demos_to_dict(demos):
    """
    Convert demos to dict that map action to a numpy array of [n_state,n_dim] where each row correspond to a state
    where the corresponding action take place
    :param demos: a list of demonstration from different experts
    :return:
    """
    n_demos = len(demos)
    demo_dict = [{} for _ in range(n_demos)]
    demos_states = [[] for _ in range(n_demos)]
    demos_actions = [[] for _ in range(n_demos)]
    for i in range(n_demos):
        for record in demos[i]:
            state = record["state"]
            action = record["action"]
            if action in demo_dict[i]:
                demo_dict[i][action].append(state)
            else:
                demo_dict[i][action] = []
                demo_dict[i][action].append(state)
            demos_states[i].append(state)
            demos_actions[i].append(action)

        for action in demo_dict[i]:
            demo_dict[i][action] = np.array(demo_dict[i][action])
        demos_states[i] = np.array(demos_states[i])
        demos_actions[i] = np.array(demos_actions[i])

    return demos_states, demos_actions, demo_dict


def calc_potential(obs, action, low, high, demo_dict, sigma=0.5):
    """
    Calculate the potential function following Brys's work
    :param obs: observation state
    :param action: action to evaluate
    :param demo_dict: dict that map action to observed states
    :param sigma: hyperparameter for covariance matrix
    :return: Phi_D(s,a)
    """
    demo_states = []
    for dd in demo_dict:
        if action in dd:
            demo_states.append(dd[action])
    demo_states = torch.cat(demo_states, 0)
    # demo_states = torch.cat([dd[action] for dd in demo_dict], 0)
    norm_demo_states = normalize_box_state(demo_states, low, high)
    norm_obs_states = normalize_box_state(obs, low, high)
    g = torch.exp(- calc_distance(norm_obs_states, norm_demo_states) / sigma)
    phi = torch.max(g)
    return phi


def calc_potential_scored(obs, action, low, high, demo_dict, score, sigma=0.5):
    """
    Calculate the potential function following Brys's work
    :param obs: observation state
    :param action: action to evaluate
    :param demo_dict: dict that map action to observed states
    :param sigma: hyperparameter for covariance matrix
    :return: Phi_D(s,a)
    """
    n_demos = len(demo_dict)
    phi = 0
    for i in range(n_demos):
        if action in demo_dict[i]:
            demo_states = demo_dict[i][action]
            norm_demo_states = normalize_box_state(demo_states, low, high)
            norm_obs_states = normalize_box_state(obs, low, high)
            g = torch.exp(- calc_distance(norm_obs_states, norm_demo_states) / sigma)
            _phi = torch.max(g)
            phi += _phi * score[i]

    return phi
