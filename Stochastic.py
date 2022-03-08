from test_env import EnvTest
from TradingDqn import testQModel, loadEnv, loadEnvNumber, Q_Model
from policy_gradient import PolicyGradient
from network_utils import device
from policy import CategoricalPolicy
from TradeEnv import TradingEnv

import matplotlib.pyplot as plt
import time
import torch
import numpy as np

class configTest():
	learning_rate 		= 0.00015
	use_baseline  		= False
	gamma 		  		= 0.99
	normalize_advantage = False
	num_batches 		= 100
	batch_size     		= 2000
	max_ep_len			= 10
	record      		= False
	scores_output 		= "scores"
	summary_freq		= 25
	state_history       = 1
	output_path 		= "results/"
	log_path     		= output_path + "log.txt"
	plot_output  		= output_path + "scores.png"
	env_name 			= "Test"

class config():
	sequence_len        = 100
	inChannels          = 8
	learning_rate 		= 0.00005
	use_baseline  		= True
	gamma 		  		= 0.99
	normalize_advantage = True
	num_batches 		= 10000 # number of batches trained on
	batch_size     		= 300 # number of steps used to compute each policy update
	max_ep_len			= 100 # maximum episode length
	record      		= False
	scores_output 		= "scores"
	summary_freq		= 25
	save_freq			= 25
	state_history       = 1
	output_path 		= "Policy/"
	log_path     		= output_path + "log.txt"
	plot_output  		= output_path + "scores.png"
	save_path 			= output_path + "model.weights"
	env_name 			= "Test"

def evaluatePolicy():
	env = loadEnv()
	net = Q_Model(env.action_space.n, config.sequence_len, config.inChannels).to(device)
	policy = CategoricalPolicy(net)
	policy.load_state_dict(torch.load(config.save_path))
	plt.clf()
	num_runs = 10
	episodes_total = 0
	for i in range(num_runs):
		print(f"Evaluating {i+1}/{num_runs}...")
		state = env.trueReset()
		total_rewards = 0
		reward_history = []
		while True:
			action = policy.act(state)

			state, reward, done, info = env.step(action)

			total_rewards += reward.sum()
			reward_history.append(total_rewards)
			if done:
				break
		plt.plot(reward_history, label=f"run {i+1}")
		episodes_total += np.array(reward_history)
	plt.plot(episodes_total/num_runs, label=f"average")
	plt.xlabel("Time")
	plt.ylabel("Reward")
	plt.legend()
	plt.savefig(config.output_path + "Evaluation.png")

def evaluateEveryEnv():
	
	net = Q_Model(3, config.sequence_len, config.inChannels).to(device)
	policy = CategoricalPolicy(net)
	policy.load_state_dict(torch.load(config.output_path + "model_1_batch.weights"))
	plt.clf()
	totalEnv = 28
	for envNum in range(totalEnv):
		env = loadEnvNumber(envNum)
		num_runs = 10
		episodes_total = 0
		print(f"Running {envNum+1}/{totalEnv}...")
		for i in range(num_runs):
			print(f"  --Evaluating {i+1}/{num_runs}...")
			state = env.trueReset()
			total_rewards = 0
			reward_history = []
			while True:
				action = policy.act(state)

				state, reward, done, info = env.step(action)

				total_rewards += reward.sum()
				reward_history.append(total_rewards)
				if done:
					break
			episodes_total += np.array(reward_history)
		plt.plot(episodes_total/num_runs, label=f"env {envNum}")
	plt.xlabel("Time")
	plt.ylabel("Reward")
	plt.legend()
	plt.savefig(config.output_path + "EnvEvaluation.png")

def main():
	env = loadEnv()
	# train model
	model = PolicyGradient(env, config, None)
	print(env.stateSpace.shape)
	start = time.time()
	model.run()
	end = time.time()
	print(f"finished in {(end-start)/60} min / {(end-start)/3600} hrs")

def test():
	rawData = np.concatenate([np.arange(1,1001).reshape(1,1000,1) for _ in range(7)], axis=-1)
	rawData = np.concatenate([rawData, np.concatenate([np.arange(1,2001,2).reshape(1,1000,1) for _ in range(7)], axis=-1)])
	env = TradingEnv(rawData, np.empty((1,10)), rawData, config.sequence_len)
	state = env.trueReset()
	print(state.shape, state[:,99,7], state[:,99,5])
	state, reward, done, info = env.step(np.array([1,0]))
	print(state[:,98:,7], reward)
	state, reward, done, info = env.step(np.array([1,1]))
	print(state[:,97:,7], reward)
	state, reward, done, info = env.step(np.array([0,1]))
	print(state[:,96:,7], reward)
	state, reward, done, info = env.step(np.array([0,0]))
	print(state[:,95:,7], reward)
	state, reward, done, info = env.step(np.array([0,0]))
	print(state[:,95:,7], reward)
	state, reward, done, info = env.step(np.array([0,1]))
	print(state[:,96:,7], reward)
	state, reward, done, info = env.step(np.array([1,0]))
	print(state[:,95:,7], reward)

if __name__ == '__main__':
	#main()
	#evaluatePolicy()
	evaluateEveryEnv()
	#test()