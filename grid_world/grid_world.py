import gym
import gym_minigrid
import numpy as np
import random
from q_learning_agent import QLearningAgent
from monte_carlo_agent import MonteCarloAgent
from value_iteration_agent import ValueIterationAgent

# Actions
# 0 - Rotate Left
# 1 - Rotate Right
# 2 - Move Forward

# Directions
# 0 - Facing Right
# 1 - Facing Down
# 2 - Facing Left
# 3 - Facing Up

# New State is a dictionary with fields
# image: partially observable view of the environment
# mission: string describing the objective for the agen
# direction: optional compass

# Mapping between positions and their corresponding state
# (x,y,direction): state
state_dict = {
  (1,1,0): 0,
  (1,1,1): 1,
  (1,1,2): 2,
  (1,1,3): 3,
  (2,1,0): 4,
  (2,1,1): 5,
  (2,1,2): 6,
  (2,1,3): 7,
  (3,1,0): 8,
  (3,1,1): 9,
  (3,1,2): 10,
  (3,1,3): 11,
  (1,2,0): 12,
  (1,2,1): 13,
  (1,2,2): 14,
  (1,2,3): 15,
  (2,2,0): 16,
  (2,2,1): 17,
  (2,2,2): 18,
  (2,2,3): 19,
  (3,2,0): 20,
  (3,2,1): 21,
  (3,2,2): 22,
  (3,2,3): 23,
  (1,3,0): 24,
  (1,3,1): 25,
  (1,3,2): 26,
  (1,3,3): 27,
  (2,3,0): 28,
  (2,3,1): 29,
  (2,3,2): 30,
  (2,3,3): 31,
  (3,3,0): 32,
  (3,3,1): 33,
  (3,3,2): 34,
  (3,3,3): 35,
}

env = gym.make('MiniGrid-Empty-5x5-v0') # Really a 3x3
action_space_size = 3
state_space_size = 36
max_steps_per_episode = 100
discount_rate = 0.99

# Uncomment the agent to test or train

# ql_agent = QLearningAgent(env, state_dict, state_space_size, action_space_size,)
# ql_agent.train()
# ql_agent.test()

# mc_agent = MonteCarloAgent(env, state_dict, state_space_size, action_space_size)
# mc_agent.train()
# mc_agent.test()

# vi_agent = ValueIterationAgent(env, state_dict, state_space_size, action_space_size)
# vi_agent.train()
# vi_agent.test()
