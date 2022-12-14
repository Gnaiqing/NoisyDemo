# NoisyDemo
Course project for CSC2626

the code requires gym and stable-baselines-3 to run properly. Install the packages using
`pip install stable-baselines3`. 

### Trajectory creation
First, train a RL agent that simulate human expert using `experts/train_model.py`. For example, train an A2C model for CartPole-v1 using stable-baselines3 by running `cd experts` then `python train_model.py --env CartPole-v1 --model A2C`. This will generate a file named `A2C_CartPole-v1.zip` under current directory.

Then create trajectory by running `experts/create_trajectory.py`. For example, to simulate trajectories from 10 users with different accuracy, run `python create_trajectory.py --dataset_path ../datasets --env CartPole-v1 --model A2C --n_experts 10 --num_noisy 5`. This will create 10 trajectories,5 from simulated experts and 5 from simulated noisy demonstrators. 

### Run reinforcement learning with demonstration
To run pure reinforcement learning using DQN, first run `cd ../rlfd` then run `python dqn.py --env CartPole-v1 --n_episodes 1000 --figpath ../fig`. This gives results for DQN (without using demonstration) in `fig` repository. 

To run RlfD without consider which user the trajectories come from, run `python rlfd.py --env CartPole-v1 --expert_label A2C --toggle_new_method 0`. To run To run our method, run `python rlfd.py --env CartPole-v1 --expert_label A2C --toggle_new_method 1`. This gives results for rlfd in `fig` repository. 



