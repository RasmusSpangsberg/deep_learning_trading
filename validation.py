import gym
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK
import matplotlib.pyplot as plt
from model import PolicyNet


env = gym.make('forex-v0',
               df = FOREX_EURUSD_1H_ASK,
               window_size = 20,
               frame_bound = (20, 300),
               unit_side = 'right')

policy = PolicyNet(env)

losses = []
training_rewards = []

observation = env.reset()
while True:
    action = policy(observation)
    observation, reward, done, info = env.step(action)

    # policy gradient update
    policy.optimizer.zero_grad()
    a_probs = policy(observation)
    loss = policy.loss(a_probs, reward)
    loss.backward()
    policy.optimizer.step()

    training_rewards.append(reward)
    losses.append(loss.item())

    if done:
        print("info:", info)
        break

print("max possible profit:", env.max_possible_profit())
print("losses:", losses)
print("training rewards:", training_rewards)
plt.cla()
env.render_all()
plt.show()