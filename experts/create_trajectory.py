import argparse
import os
import gym
import numpy as np
import json
from stable_baselines3 import DQN, A2C, PPO
from pathlib import Path
from train_model import train_model

def to_file(x):
    """
    Convert a point x from openai gym space to serializable format
    :param x:
    :return:
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, int):
        return x
    else:
        raise NotImplementedError

def to_point(x):
    """
    Convert a point x from serailizable format to point in gym space
    :param x:
    :return:
    """
    if isinstance(x, list):
        return np.array(x)
    elif isinstance(x, int):
        return x
    else:
        raise NotImplementedError


def collect_trajectory(model, env, noise=0.0, timesteps=1000):
    obs = env.reset()
    trajectory = []
    for i in range(timesteps):
        last_obs = obs
        action, _states = model.predict(obs, deterministic=True)
        p = np.random.rand()
        if p < noise:
            action = env.action_space.sample()

        obs, rewards, done, info = env.step(action)
        if done:
            obs = env.reset()
        trajectory.append({
            "state": to_file(last_obs),
            "action": to_file(action)
        })

    return trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--model", type=str, default="A2C")
    parser.add_argument("--dataset_path", type=str, default="../datasets")
    parser.add_argument("--n_experts", type=int, default=10)
    parser.add_argument("--min_noise", type=float, default=0.0)
    parser.add_argument("--max_noise", type=float, default=0.3)
    args = parser.parse_args()
    tag = f"{args.model}_{args.env}"
    env = gym.make(args.env)
    if os.path.exists(f"{tag}.zip"):
        if args.model == "DQN":
            model = DQN.load(tag)
        elif args.model == "A2C":
            model = A2C.load(tag)
        elif args.model == "PPO":
            model = PPO.load(tag)
        else:
            raise NotImplementedError
    else:
        model = train_model(args.env, args.model)

    np.random.seed(0)
    for i in range(args.n_experts):
        # noise = args.min_noise + np.random.rand() * (args.max_noise - args.min_noise)
        # noise = args.min_noise + i * (args.max_noise - args.min_noise) / (args.n_experts - 1)
        if i < args.n_experts // 2:
            noise = 0
        else:
            noise = 1
        trajectory = collect_trajectory(model, env, noise)
        trajectory_path = Path(args.dataset_path) / f"{args.model}_{args.env}_{i}.json"
        with open(trajectory_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=4)









