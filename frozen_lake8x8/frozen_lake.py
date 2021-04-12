import gym
import numpy as np
from agents.QLearningAgent import QLearningAgent
from agents.MonteCarloAgent import MonteCarloAgent
from agents.SarsaAgent import SarsaAgent
from agents.ExpectedSarsaAgent import ExpectedSarsaAgent
from agents.DoubleQLearningAgent import DoubleQLearningAgent
from agents.NStepSarsaAgent import NStepSarsaAgent
from agents.TreeBackupAgent import TreeBackupAgent

# Actions: 0 - Left, 1 - Down, 2 - Right, 3 - Up
# 33% probability of the agent taking the chosen action
# Observation: Number between 0 and 63 (inclusive)
# representing the tile the agent is at
# Rewawrd: 1 for reaching the goal, 0 otherwise

env = gym.make('FrozenLake8x8-v0')
num_training_episodes = 100000
num_testing_episodes = 100000

ql_agent = QLearningAgent()
# ql_agent.train(env, num_training_episodes)

mc_agent = MonteCarloAgent()
# mc_agent.train(env, num_training_episodes)

sarsa_agent = SarsaAgent()
# sarsa_agent.train(env, num_training_episodes)

expected_sarsa_agent = ExpectedSarsaAgent()
# expected_sarsa_agent.train(env, num_training_episodes)

double_ql_agent = DoubleQLearningAgent()
# double_ql_agent.train(env, num_training_episodes)

nstep_sarsa_agent = NStepSarsaAgent()
# nstep_sarsa_agent.train(env, 3, num_training_episodes)

tree_backup_agent = TreeBackupAgent()
# tree_backup_agent.train(env, 3, num_training_episodes)

ql_policy_100k = np.load('policies/ql_policy_100k.npy',allow_pickle='TRUE').item()
ql_policy_1mil = np.load('policies/ql_policy_1mil.npy',allow_pickle='TRUE').item()
ql_policy_10k_edr0001 = np.load('policies/ql_policy_10k_edr0001.npy',allow_pickle='TRUE').item()
ql_policy_100k_edr0001 = np.load('policies/ql_policy_100k_edr0001.npy',allow_pickle='TRUE').item()
ql_policy_100k_lr01 = np.load('policies/ql_policy_100k_lr01.npy',allow_pickle='TRUE').item()
sarsa_policy_10k_edr0001 = np.load('policies/sarsa_policy_10k_edr0001.npy',allow_pickle='TRUE').item()
sarsa_policy_100k_edr0001 = np.load('policies/sarsa_policy_100k_edr0001.npy',allow_pickle='TRUE').item()
mc_policy_1k = np.load('policies/mc_policy_1k.npy',allow_pickle='TRUE').item()
mc_policy_10k = np.load('policies/mc_policy_10k.npy',allow_pickle='TRUE').item()
mc_policy_100k = np.load('policies/mc_policy_100k.npy',allow_pickle='TRUE').item()
expected_sarsa_policy_10k = np.load('policies/expected_sarsa_policy_10k.npy',allow_pickle='TRUE').item()
expected_sarsa_policy_100k = np.load('policies/expected_sarsa_policy_100k.npy',allow_pickle='TRUE').item()
expected_sarsa_policy_100k_min03 = np.load('policies/expected_sarsa_policy_100k_min03.npy',allow_pickle='TRUE').item()
dql_policy_10k = np.load('policies/dql_policy_10k.npy',allow_pickle='TRUE').item()
dql_policy_100k = np.load('policies/dql_policy_100k.npy',allow_pickle='TRUE').item()
three_step_sarsa_policy_10k = np.load('policies/3_step_sarsa_policy_10k.npy',allow_pickle='TRUE').item()
three_step_sarsa_policy_100k = np.load('policies/3_step_sarsa_policy_100k.npy',allow_pickle='TRUE').item()
four_step_sarsa_policy_10k = np.load('policies/4_step_sarsa_policy_10k.npy',allow_pickle='TRUE').item()
four_step_sarsa_policy_100k = np.load('policies/4_step_sarsa_policy_100k.npy',allow_pickle='TRUE').item()
three_step_tree_backup_policy_10k = np.load('policies/3_step_tree_backup_policy_10k.npy',allow_pickle='TRUE').item()
three_step_tree_backup_policy_100k = np.load('policies/3_step_tree_backup_policy_100k.npy',allow_pickle='TRUE').item()


# ql_agent.test(env, num_testing_episodes, ql_policy_100k_lr01)
# ql_agent.testVisual(env, 3, ql_policy_100k_edr0001)
# sarsa_agent.test(env, num_testing_episodes, sarsa_policy_100k_edr0001)
# mc_agent.test(env, num_testing_episodes, mc_policy_100k)
# expected_sarsa_agent.test(env, num_testing_episodes, expected_sarsa_policy_100k_min03)
# double_ql_agent.test(env, num_testing_episodes, dql_policy_100k)
# nstep_sarsa_agent.test(env, num_testing_episodes, three_step_sarsa_policy_100k)
tree_backup_agent.test(env,num_testing_episodes, three_step_tree_backup_policy_100k)

def printPolicy(policy):
	start = 0
	arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
	for i in range(8):
		row = ''
		for j in range(start, start+8):
			row += arrows[policy[j]] + ' '
		print(row)
		start += 8

# printPolicy(ql_policy_100k_edr0001)
# print()
# printPolicy(dql_policy_100k)