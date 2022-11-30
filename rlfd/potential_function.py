import gym
import json
import numpy as np
from pathlib import Path


def normalize_box_state(obs, low, high):
    """
    Normalize observation to range [0,1]
    :param obs: observation or list of observations of shape (n_states, n_dim)
    :param low: lowest bound of space
    :param high: highest bound of space
    :return: x: normalized value in [0,1]
    """
    obs = np.array(obs)
    x = (obs - low) / (high - low)
    return x


def calc_distance(x1, x2, metric="l2"):
    if metric == "l2":
        return np.sum((x2-x1)**2, axis=1)
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
    demo_dict = {}
    for demo in demos:
        for record in demo:
            state = record["state"]
            action = record["action"]
            if action in demo_dict:
                demo_dict[action].append(state)
            else:
                demo_dict[action] = []
                demo_dict[action].append(state)

    for action in demo_dict:
        demo_dict[action] = np.array(demo_dict[action])

    return demo_dict


def calc_potential(obs, action, low, high, demo_dict, sigma=0.5):
    """
    Calculate the potential function following Brys's work
    :param obs: observation state
    :param action: action to evaluate
    :param demo_dict: dict that map action to observed states
    :param sigma: hyperparameter for covariance matrix
    :return: Phi_D(s,a)
    """
    demo_states = demo_dict[action]
    norm_demo_states = normalize_box_state(demo_states, low, high)
    norm_obs_states = normalize_box_state(obs, low, high)
    g = np.exp(- calc_distance(norm_obs_states, norm_demo_states) / sigma)
    phi = np.max(g)
    return phi


