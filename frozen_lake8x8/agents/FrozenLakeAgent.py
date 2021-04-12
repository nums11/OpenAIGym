import random
import numpy as np
from tqdm import tqdm
import collections
from collections import Counter
import matplotlib.pyplot as plt
import time

class FrozenLakeAgent:
  def __init__(self):
    self.state_space_size = 64
    self.action_space_size = 4

  def printRewards(self, split, num_episodes, rewards):
    rewards_per_split_episodes = np.split(np.array(rewards),num_episodes/split)
    count = split

    print(f"\n********Average reward per {split} episodes ********\n")
    for r in rewards_per_split_episodes:
      print(count, ": ", str(sum(r/split)))
      count += split

  def savePolicyFromQTable(self, q_table, policy_name):
    policy = self.getPolicyFromQTable(q_table)
    np.save('policies/' + policy_name + '.npy', policy)

  def getPolicyFromQTable(self, q_table):
    policy = {}
    for i in range(len(q_table)):
      policy[i] = np.argmax(q_table[i])
    return policy

  def test(self, env, num_episodes, policy):
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
    print(f'******* Average rewards across {num_episodes} *******')
    print(avg_rewards)
    counts = Counter(rewards_all_episodes)
    win_rate = (counts[1] / len(rewards_all_episodes)) * 100
    print(f'{win_rate}% win rate')


  def testVisual(self, env, num_episodes, policy):
    for episode in range(num_episodes):
      print("*****EPISODE ", episode+1, "*****\n")
      time.sleep(1)

      state = env.reset()
      done = False
      while True:       
        env.render()
        time.sleep(0.1)
        action = policy[state]
        new_state, reward, done, info = env.step(action)
        state = new_state

        if done:
          env.render()
          if reward == 1:
            print("****You reached the goal!****\n")
            time.sleep(3)
          else:
            print("****You fell through a hole!****\n")
            time.sleep(3)
          break

    env.close()