import gym
import numpy as np
from tqdm import tqdm
import random
from simple_q_learning_agent import SimpleQLearningAgent
from advanced_q_learning_agent import AdvancedBlackjackAgent
from simple_monte_carlo_agent import SimpleMonteCarloAgent
from advanced_monte_carlo_agent import AdvancedMonteCarloAgent

def comparePolicies(p1, p2):
  print('State        P1 Action   P2 Action   Different?')
  for state, action in p1.items():
    diff_string = ''
    if action != p2[state]:
      diff_string = '        DIFFERENT'
    print(f'{state}:    {action}            {p2[state]}'
      + diff_string)

# Actions: 0 - stick, 1 - hit
# Observation: 
# 0 - Player's current sum
# 1 - The dealer's one showing card
# 2 - Whether or not the player holds a usable ace (0 or 1)

env = gym.make('Blackjack-v0')
state_space_size = 2
action_space_size = 2
num_episodes = 1000000

simple_ql_agent = SimpleQLearningAgent()
# simple_ql_agent.train(env, num_episodes)

# advanced_agent = AdvancedBlackjackAgent()
# advanced_agent.train(env, num_episodes)

simple_monte_carlo_agent = SimpleMonteCarloAgent()
# simple_monte_carlo_agent.train(env, num_episodes)

advanced_monte_carlo_agent = AdvancedMonteCarloAgent()
# advanced_monte_carlo_agent.train(env,num_episodes)

simple_mc_policy_100 = np.load('simple_mc_policy_100.npy',allow_pickle='TRUE').item()
simple_mc_policy_1k = np.load('simple_mc_policy_1k.npy',allow_pickle='TRUE').item()
simple_mc_policy_10k = np.load('simple_mc_policy_10k.npy',allow_pickle='TRUE').item()
simple_mc_policy_100k = np.load('simple_mc_policy_100k.npy',allow_pickle='TRUE').item()
simple_mc_policy_1mil = np.load('simple_mc_policy_1mil.npy',allow_pickle='TRUE').item()
advanced_mc_policy_100 = np.load('advanced_mc_policy_100.npy',allow_pickle='TRUE').item()
advanced_mc_policy_1k = np.load('advanced_mc_policy_1k.npy',allow_pickle='TRUE').item()
advanced_mc_policy_10k = np.load('advanced_mc_policy_10k.npy',allow_pickle='TRUE').item()
advanced_mc_policy_100k = np.load('advanced_mc_policy_100k.npy',allow_pickle='TRUE').item()
advanced_mc_policy_1mil = np.load('advanced_mc_policy_1mil.npy',allow_pickle='TRUE').item()

# simple_ql_policy = np.load('simple_ql_policy.npy',allow_pickle='TRUE').item()

# comparePolicies(simple_mc_policy_1k, simple_mc_policy_1mil)

# simple_ql_agent.test(env, 100000, simple_mc_policy_1mil)
simple_monte_carlo_agent.test(env, 1000000, simple_mc_policy_1mil)
# advanced_monte_carlo_agent.test(env, 1000000, advanced_mc_policy_1k)

# rewards = []
# with open('advanced_rewards') as f:
#   rewards = f.read().splitlines()
# rewards = np.array(rewards)
# rewards = [float(i) for i in rewards]
# split = 100000
# rewards_per_split_episodes = np.split(np.array(rewards),num_episodes/split)
# count = split

# print(f"\n********Average reward per {split} episodes ********\n")
# for r in rewards_per_split_episodes:
#   print(count, ": ", str(sum(r/split)))
#   count += split