import gym
import numpy as np
from tqdm import tqdm
import random
from agents.simple_q_learning_agent import SimpleQLearningAgent
from agents.advanced_q_learning_agent import AdvancedQLearningjackAgent
from agents.simple_monte_carlo_agent import SimpleMonteCarloAgent
from agents.advanced_monte_carlo_agent import AdvancedMonteCarloAgent
from agents.simple_sarsa_agent import SimpleSarsaAgent
from agents.advanced_sarsa_agent import AdvancedSarsaAgent

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

# simple_ql_agent = SimpleQLearningAgent()
# simple_ql_agent.train(env, num_episodes)

advanced_ql_agent = AdvancedQLearningjackAgent()
# advanced_ql_agent.train(env, num_episodes)

simple_monte_carlo_agent = SimpleMonteCarloAgent()
# simple_monte_carlo_agent.train(env, num_episodes)

advanced_monte_carlo_agent = AdvancedMonteCarloAgent()
# advanced_monte_carlo_agent.train(env,num_episodes)

# simple_sarsa_agent = SimpleSarsaAgent()
# simple_sarsa_agent.train(env, num_episodes)

advanced_sarsa_agent = AdvancedSarsaAgent()
# advanced_sarsa_agent.train(env, num_episodes)

simple_mc_policy_100 = np.load('policies/simple_mc_policy_100.npy',allow_pickle='TRUE').item()
simple_mc_policy_1k = np.load('policies/simple_mc_policy_1k.npy',allow_pickle='TRUE').item()
simple_mc_policy_1k_disc_09 = np.load('policies/simple_mc_policy_1k_disc_09.npy',allow_pickle='TRUE').item()
simple_mc_policy_1k_disc_095 = np.load('policies/simple_mc_policy_1k_disc_095.npy',allow_pickle='TRUE').item()
simple_mc_policy_10k = np.load('policies/simple_mc_policy_10k.npy',allow_pickle='TRUE').item()
simple_mc_policy_10k_disc_095 = np.load('policies/simple_mc_policy_10k_disc_095.npy',allow_pickle='TRUE').item()
simple_mc_policy_100k = np.load('policies/simple_mc_policy_100k.npy',allow_pickle='TRUE').item()
simple_mc_policy_1mil = np.load('policies/simple_mc_policy_1mil.npy',allow_pickle='TRUE').item()
advanced_mc_policy_100 = np.load('policies/advanced_mc_policy_100.npy',allow_pickle='TRUE').item()
advanced_mc_policy_1k = np.load('policies/advanced_mc_policy_1k.npy',allow_pickle='TRUE').item()
advanced_mc_policy_10k = np.load('policies/advanced_mc_policy_10k.npy',allow_pickle='TRUE').item()
advanced_mc_policy_100k = np.load('policies/advanced_mc_policy_100k.npy',allow_pickle='TRUE').item()
advanced_mc_policy_1mil = np.load('policies/advanced_mc_policy_1mil.npy',allow_pickle='TRUE').item()
custom_policy = np.load('policies/custom_policy.npy',allow_pickle='TRUE').item()
custom_policy_two = np.load('policies/custom_policy_two.npy',allow_pickle='TRUE').item()
custom_policy_three = np.load('policies/custom_policy_three.npy',allow_pickle='TRUE').item()
custom_policy_four = np.load('policies/custom_policy_four.npy',allow_pickle='TRUE').item()
articl_policy = np.load('policies/articl_policy.npy',allow_pickle='TRUE').item()
simple_ql_policy = np.load('policies/simple_ql_policy.npy',allow_pickle='TRUE').item()
simple_ql_policy_two = np.load('policies/simple_ql_policy_two.npy',allow_pickle='TRUE').item()
advanced_ql_policy_two = np.load('policies/advanced_ql_policy_two.npy',allow_pickle='TRUE').item()
advanced_ql_policy_three = np.load('policies/advanced_ql_policy_three.npy',allow_pickle='TRUE').item()
simple_sarsa_policy = np.load('policies/simple_sarsa_policy.npy',allow_pickle='TRUE').item()
advanced_sarsa_policy = np.load('policies/advanced_sarsa_policy.npy',allow_pickle='TRUE').item()
# print("custom_policy_four", custom_policy_four)

# simple_ql_policy = np.load('simple_ql_policy.npy',allow_pickle='TRUE').item()

# comparePolicies(advanced_ql_policy_two, advanced_ql_policy_three)

# simple_ql_agent.test(env, 100000, simple_ql_policy)
# advanced_ql_agent.test(env, 100000, advanced_ql_policy_three)
# simple_monte_carlo_agent.test(env, 1000000, simple_mc_policy_10k_disc_095)
# advanced_monte_carlo_agent.test(env, 1000000, articl_policy)
# simple_sarsa_agent.test(env, 100000, simple_sarsa_policy)
advanced_sarsa_agent.test(env, 100000, advanced_sarsa_policy)


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