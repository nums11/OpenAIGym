import numpy as np
import gym
import random
import time
from IPython.display import clear_output
from gym.utils.play import play
from tqdm import tqdm
from collections import Counter
from q_table import q_table
q_table = np.array(q_table)

env = gym.make("FrozenLake-v0")
# env.reset()
# print(env.step(2))
# env.render()
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

max_steps_per_episode = 100

def train():
  q_table = np.zeros((state_space_size, action_space_size))
  learning_rate = 0.1
  discount_rate = 0.99
  exploration_rate = 1
  max_exploration_rate = 1
  min_exploration_rate = 0.01
  exploration_decay_rate = 0.001
  num_episodes = 10000
  rewards_all_episodes = []
  # Q-Learning Algorithm (Training)
  for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
      # Exploration-exploitation trade-off
      exploration_rate_threshold = random.uniform(0, 1)
      if exploration_rate_threshold > exploration_rate:
        action = np.argmax(q_table[state,:]) 
      else:
        action = env.action_space.sample()

      new_state, reward, done, info = env.step(action)
      # Update Q-table for Q(s,a)
      q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
        learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
      state = new_state
      rewards_current_episode += reward 

    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
      (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    rewards_all_episodes.append(rewards_current_episode)


  # Calculate and print the average reward per thousand episodes
  rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
  count = 1000

  print("********Average reward per thousand episodes********\n")
  for r in rewards_per_thousand_episodes:
      print(count, ": ", str(sum(r/1000)))
      count += 1000

  print("\n\n********Q-table********\n")
  print(q_table)

def getPolicyFromQTable(q_table):
  policy = {}
  for i in range(len(q_table)):
    policy[i] = np.argmax(q_table[i])
  return policy

def testVisual():
  for episode in range(3):
    state = env.reset()
    done = False
    print("*****EPISODE ", episode+1, "*****\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):        
      env.render()
      time.sleep(0.3)
      action = np.argmax(q_table[state,:])        
      new_state, reward, done, info = env.step(action)

      if done:
        env.render()
        if reward == 1:
          print("****You reached the goal!****\n")
          time.sleep(3)
        else:
          print("****You fell through a hole!****\n")
          time.sleep(3)
        break

      state = new_state

  env.close()

def test(num_episodes, policy):
  done = False
  rewards_all_episodes = []

  for episode in tqdm(range(num_episodes)):
    state = env.reset()
    done = False
    rewards_current_episode = 0

    while True:
      if done:
        rewards_all_episodes.append(rewards_current_episode)
        break
      action = policy[state]
      new_state, reward, done, info = env.step(action)
      rewards_current_episode += reward
      state = new_state

  avg_rewards = sum(rewards_all_episodes) / num_episodes
  print(f'******* Average rewards across {num_episodes} episodes *******')
  print(avg_rewards)
  counts = Counter(rewards_all_episodes)
  win_rate = (counts[1] / len(rewards_all_episodes)) * 100
  print(f'{win_rate}% win rate')


# train()
# testVisual()
policy = getPolicyFromQTable(q_table)
test(100000, policy)

