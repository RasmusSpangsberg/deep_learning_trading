import gym
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK
import matplotlib.pyplot as plt
from model import Model


env = gym.make('forex-v0',
               df = FOREX_EURUSD_1H_ASK,
               window_size = 10,
               frame_bound = (10, 300),
               unit_side = 'right')

model = Model(env)


observation = env.reset()
while True:
    action = model.predict(observation)
    observation, reward, done, info = env.step(action)

    model.fit(reward)

    if done:
        print("info:", info)
        break

print("max possible profit:", env.max_possible_profit())
plt.cla()
env.render_all()
plt.show()