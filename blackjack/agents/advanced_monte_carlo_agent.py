import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from collections import Counter

"""
State space includes players sum, dealers sum,
and whether or not the player has a usable ace
State-space size: 420
"""
class AdvancedMonteCarloAgent:
  def train(self, env, num_episodes):
    state_space_size = 2
    action_space_size = 2
    # learning_rate = 0.1
    # learning_rate = 0.05
    learning_rate = 0.01
    rewards_all_episodes = []
    # discount_rate = 0.99
    discount_rate = 1
    q_table = {}
    bool = False
    for player_sum in range(1,22):
      for dealer_sum in range(1, 11):
        q_table[(player_sum, dealer_sum, bool)] = np.zeros((action_space_size))
        bool = not bool
        q_table[((player_sum, dealer_sum, bool))] = np.zeros((action_space_size))
        bool = not bool

    # Holds the returns for every state action pair across all episodes
    returns = {}
    # Holds the deltas for every state action pair across all episodes
    deltas = {}
    for key in q_table.keys():
      for action in range(2):
        returns[(key,action)] = list()
        deltas[(key,action)] = list()

    for _ in tqdm(range(num_episodes)):
      episode = self.generateBlackJackEpisode(env)
      G = 0
      for i, step in enumerate(episode[::-1]): # reverse the list
        # Every step is represented as an array with 4 elements:
        # initial state, action, reward, new_state
        G = discount_rate * G + step[2]
        initial_state = step[0]
        action = step[1]
        # If this step was not already visited in the episode
        if initial_state not in [x[0] for x in episode[::-1][len(episode)-i:]]:
          returns[(initial_state, action)].append(G)
          new_avg_state_action_value = np.average(returns[(initial_state, action)])
          delta = np.abs(q_table[initial_state][action] - new_avg_state_action_value)
          deltas[(initial_state, action)].append(delta)
          q_table[initial_state][action] = new_avg_state_action_value

    policy = self.getPolicyFromQTable(q_table)
    self.savePolicy(policy)
    # Plot change in deltas over iterations
    plt.figure(figsize=(20,10))
    all_series = [list(x)[:50] for x in deltas.values()]
    for series in all_series:
      plt.plot(series)
    plt.show()

  def generateBlackJackEpisode(self, env):
    player_sum, dealer_sum, has_ace = env.reset()
    state = (player_sum, dealer_sum, has_ace)
    # 2-d array where each element is a step in the episode
    # represented as an array with 4 elements:
    # initial state, action, reward, new_state
    episode = []
    done = False

    while True:
      if done:
        return episode
      action = self.getRandomAction()
      observation, reward, done, info = env.step(action)
      player_sum, dealer_sum, has_ace = observation
      new_state = (player_sum,dealer_sum, has_ace)
      episode.append([state, action, reward, new_state])
      state = new_state

  # Return random int between 0 and 1 (inclusive)
  def getRandomAction(self):
    return random.randint(0,1)

  def printRewards(self, num_episodes, rewards):
    split = 100
    rewards_per_split_episodes = np.split(np.array(rewards),num_episodes/split)
    count = split

    print(f"\n********Average reward per {split} episodes ********\n")
    for r in rewards_per_split_episodes:
      print(count, ": ", str(sum(r/split)))
      count += split

  def saveRewards(self, rewards):
    with open('simple_rewards', 'w') as file_handler:
      for item in rewards:
        file_handler.write("{}\n".format(item))

  def getPolicyFromQTable(self, q_table):
    policy = {key:np.argmax(action_values) for key, action_values in q_table.items()}
    return policy

  def savePolicy(self, policy):
    np.save('policy.npy', policy)

  def test(self, env, num_episodes, policy):
    env.reset()
    done = False
    rewards_all_episodes = []

    num_skipped = 0
    for episode in tqdm(range(num_episodes)):
      player_sum, dealer_sum, has_ace = env.reset()
      # Special case - when you auto win
      if player_sum == 21:
        num_skipped += 1
        continue

      state = (player_sum, dealer_sum, has_ace)
      done = False
      while True:
        if done:
          rewards_all_episodes.append(reward)
          break
        action = policy[state]
        observation, reward, done, info = env.step(action)
        player_sum, dealer_sum, has_ace = observation
        new_state = (player_sum, dealer_sum, has_ace)
        state = new_state

    avg_rewards = sum(rewards_all_episodes) / (num_episodes - num_skipped)
    print(f'******* Average rewards across {num_episodes} episodes ({num_skipped}) skipped *******')
    print(avg_rewards)
    counts = Counter(rewards_all_episodes)
    win_rate = (counts[1] / len(rewards_all_episodes)) * 100
    print(f'{win_rate}% win rate')


