import torch


class Model:
	def __init__(self, env):
		self.env = env

	def predict(self, observation):
		return self.env.action_space.sample()

	def fit(self, reward):
		pass