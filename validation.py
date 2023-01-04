import gym
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt
from model import PolicyNet
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import pandas as pd
import torch

torch.manual_seed(0)

# data
csv_filename = "STOCK_US_XNYS_GME.csv"
df = pd.read_csv(csv_filename)
df.set_index('Date', inplace=True)

# training
env = gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)
env = DummyVecEnv([lambda: env])

policy = A2C("MlpPolicy", env)
policy.learn(total_timesteps=10000)

# evaluation
env = gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)

observation = env.reset()
while True:
    observation = observation[np.newaxis, ...]
    action, _states = policy.predict(observation)
    #action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        print("info:", info)
        break

print("max possible profit:", env.max_possible_profit())
plt.cla()
env.render_all()
plt.show()