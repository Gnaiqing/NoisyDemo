import argparse
import gym
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy


def train_model(env, model_name):
    if model_name == "A2C":
        model = A2C("MlpPolicy", env)
    elif model_name == "DQN":
        model = DQN("MlpPolicy", env)
    elif model_name == "PPO":
        model = PPO("MlpPolicy", env)
    else:
        raise NotImplementedError

    model.learn(total_timesteps=int(1e5), progress_bar=True)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--model", type=str, default="A2C")
    args = parser.parse_args()
    env = gym.make(args.env)
    model = train_model(env, args.model)
    tag = f"{args.model}_{args.env}"
    model.save(tag)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Mean reward:{mean_reward:.3f}")
    print(f"Std reward:{std_reward:.3f}")

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


