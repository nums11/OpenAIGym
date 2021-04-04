import gym
import numpy as np
from tqdm import tqdm
import random
from simple_blackjack_agent import SimpleBlackjackAgent
from advanced_blackjack_agent import AdvancedBlackjackAgent

# Actions: 0 - stick, 1 - hit
# Observation: 
# 0 - Player's current sum
# 1 - The dealer's one showing card
# 2 - Whether or not the player holds a usable ace (0 or 1)

env = gym.make('Blackjack-v0')
state_space_size = 2
action_space_size = 2
num_episodes = 1000000

# simple_agent = SimpleBlackjackAgent()
# simple_agent.train(env, num_episodes)

# advanced_agent = AdvancedBlackjackAgent()
# advanced_agent.train(env, num_episodes)

rewards = []
with open('rewards') as f:
  rewards = f.read().splitlines()
rewards = np.array(rewards)
rewards = [float(i) for i in rewards]
split = 100000  # 100k
rewards_per_split_episodes = np.split(np.array(rewards),num_episodes/split)
count = split

print(f"\n********Average reward per {split} episodes ********\n")
for r in rewards_per_split_episodes:
  print(count, ": ", str(sum(r/split)))
  count += split