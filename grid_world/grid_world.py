import gym
import gym_minigrid
import time
import numpy as np
import random
# Optimal Q Table found after training
from q_table import q_table
q_table = np.array(q_table)

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

env = gym.make('MiniGrid-Empty-5x5-v0') # Really a 3x3
action_space_size = 3
state_space_size = 36
max_steps_per_episode = 100

# Gets unique index for one of the 36 possible states
# based on position and direction
def getState():
  x,y = env.agent_pos
  direction = env.agent_dir
  if x == 1 and y == 1:
    return 0 + direction # states 0 - 3
  elif x == 2 and y == 1:
    return 4 + direction # states 4 - 7
  elif x == 3 and y == 1:
    return 8 + direction # states 8 - 11
  elif x == 1 and y == 2:
    return 12 + direction # states 12 - 15
  elif x == 2 and y == 2:
    return 16 + direction # states 16 - 19
  elif x == 3 and y == 2:
    return 20 + direction # states 20 - 23
  elif x == 1 and y == 3:
    return 24 + direction # states 24 - 27
  elif x == 2 and y == 3:
    return 28 + direction # states 28 - 31
  elif x == 3 and y == 3:
    return 32 + direction # states 32 - 35

# Return random number between 0 and 2 (inclusive)
def getRandomAction():
  return random.randint(0,2)

def train_q_learning():
  q_table = np.zeros((state_space_size, action_space_size))
  learning_rate = 0.1
  discount_rate = 0.99
  epsilon = 1
  min_epsilon = 0.01
  epsilon_decay_rate = 0.001
  num_episodes = 10000
  rewards_all_episodes = []

  for episode in range(num_episodes):
    env.reset()
    done = False
    rewards_current_episode = 0
    print("episode ", episode)

    for step in range(max_steps_per_episode):
      state = getState()

      exploration_rate_threshold = random.uniform(0, 1)
      if exploration_rate_threshold > epsilon:
        action = np.argmax(q_table[state,:]) 
      else:
        action = getRandomAction()

      new_state, reward, done, info = env.step(action)
      new_state = getState()
      # Update Q-table for Q(s,a)
      q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
        learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

      rewards_current_episode += reward

      if done:
        print("REACHED GOAL --------------------------------")
        break

    # Exponentially decay epsilon
    epsilon = min_epsilon + \
      (1 - min_epsilon) * np.exp(-epsilon_decay_rate*episode)
    rewards_all_episodes.append(rewards_current_episode)

    print("Rewards current episode:", rewards_current_episode)

  # Calculate and print the average reward per thousand episodes
  rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
  count = 1000

  print("\n********Average reward per thousand episodes********\n")
  for r in rewards_per_thousand_episodes:
      print(count, ": ", str(sum(r/1000)))
      count += 1000

  print("\n\n********Q-table********\n")
  print(q_table)

def test():
  env.reset()
  done = False

  for step in range(max_steps_per_episode):        
    env.render()
    time.sleep(0.3)
    state = getState()
    action = np.argmax(q_table[state,:])
    new_state, reward, done, info = env.step(action)

    if done:
      env.render()
      time.sleep(10)
      break

  env.close()

# train_q_learning()
test()