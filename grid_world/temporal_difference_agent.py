import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
# Optimal Q Tables found after training
from q_tables import *

class TemporalDifferenceAgent:
  def __init__(self, env, state_dict, state_space_size, action_space_size):
    self.env = env
    self.state_dict = state_dict
    self.state_space_size = state_space_size
    self.action_space_size = action_space_size

  def train_q_learning(self):
    learning_rate = 0.1
    epsilon = 1
    min_epsilon = 0.01
    epsilon_decay_rate = 0.001
    num_episodes = 10000
    rewards_all_episodes = []
    max_steps_per_episode = 100
    discount_rate = 0.99
    q_table = np.zeros((self.state_space_size, self.action_space_size))

    for episode in tqdm(range(num_episodes)):
      self.env.reset()
      done = False
      rewards_current_episode = 0

      for step in range(max_steps_per_episode):
        state = self.getState()

        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > epsilon:
          action = np.argmax(q_table[state,:]) 
        else:
          action = self.getRandomAction()

        new_state, reward, done, info = self.env.step(action)
        new_state = self.getState()
        # Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
          learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        rewards_current_episode += reward

        if done:
          break

      # Exponentially decay epsilon
      epsilon = min_epsilon + \
        (1 - min_epsilon) * np.exp(-epsilon_decay_rate*episode)
      rewards_all_episodes.append(rewards_current_episode)

    # Calculate and print the average reward per thousand episodes
    rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
    count = 1000

    print("\n********Average reward per thousand episodes********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r/1000)))
        count += 1000

    print("\n\n********Q-table********\n")
    print(q_table)

  def train_td0(self):
    num_episodes = 10000
    discount_factor = 0.1
    learning_rate = 0.1
    # Holds the returns for every state across all episodes
    returns = {key:list() for key in self.state_dict.keys()}
    # Holds the deltas in state value for every state across all episodes
    deltas = {key:list() for key in self.state_dict.keys()}
    # Holds the state values (initially 0 for all states)
    V = {key:0 for key in self.state_dict.keys()}

    for i in tqdm(range(num_episodes)):
      self.generateEpisode(V, discount_factor, learning_rate, deltas)

    print("State Value function", V)
    print("\n")
    policy = self.getPolicyFromStateValueFunction(V)
    print('Policy', policy)
    # Plot change in deltas over iterations
    plt.figure(figsize=(20,10))
    all_series = [list(x)[:50] for x in deltas.values()]
    for series in all_series:
      plt.plot(series)
    plt.show()

  def train_sarsa(self):
    num_episodes = 10000
    learning_rate = 0.1
    discount_factor = 0.1
    epsilon = 1
    min_epsilon = 0.01
    epsilon_decay_rate = 0.001
    q_table = np.zeros((self.state_space_size, self.action_space_size))
    max_steps_per_episode = 100
    rewards_all_episodes = []

    for episode in tqdm(range(num_episodes)):
      rewards_current_episode = 0
      self.env.reset()
      state = self.getState()
      action = self.chooseSarsaAction(q_table, state, epsilon)

      for step in range(max_steps_per_episode):
        if state >= 32: # reached goal
          break

        observation, reward, done, info = self.env.step(action)
        new_state = self.getState()
        next_action = self.chooseSarsaAction(q_table, state, epsilon)
        # Update Q-table for Q(s,a)
        q_table[state, action] += learning_rate * (
          reward + discount_factor * q_table[new_state, next_action] - q_table[state,action])

        state = new_state
        action = next_action

        rewards_current_episode += reward

      # Exponentially decay epsilon
      epsilon = min_epsilon + \
        (1 - min_epsilon) * np.exp(-epsilon_decay_rate*episode)
      rewards_all_episodes.append(rewards_current_episode)

    # Calculate and print the average reward per thousand episodes
    rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
    count = 1000

    print("\n********Average reward per thousand episodes********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r/1000)))
        count += 1000

    print("\n\n********Q-table********\n")
    print(q_table)

    # Plot change in deltas over iterations
    plt.figure(figsize=(20,10))
    # all_series = [list(x)[:50] for x in deltas.values()]
    for rewards in rewards_per_thousand_episodes:
      plt.plot(rewards)
    plt.show()

  def train_expected_sarsa(self):
    num_episodes = 10000
    learning_rate = 0.1
    discount_factor = 0.1
    epsilon = 1
    min_epsilon = 0.01
    epsilon_decay_rate = 0.001
    q_table = np.zeros((self.state_space_size, self.action_space_size))
    max_steps_per_episode = 100
    rewards_all_episodes = []

    for episode in tqdm(range(num_episodes)):
      rewards_current_episode = 0
      self.env.reset()
      state = self.getState()
      action = self.chooseSarsaAction(q_table, state, epsilon)

      for step in range(max_steps_per_episode):
        if state >= 32: # reached goal
          break

        predict = q_table[state,action]
        observation, reward, done, info = self.env.step(action)
        new_state = self.getState()

        max_q = np.max(q_table[new_state,:])
        num_greedy_actions = 0
        for action in range(self.action_space_size):
          if q_table[new_state,action] == max_q:
            num_greedy_actions += 1
        non_greedy_action_prob = epsilon / self.action_space_size
        greedy_action_prob = ((1 - epsilon) / num_greedy_actions) + non_greedy_action_prob

        expected_q = 0
        for action in range(self.action_space_size):
          if q_table[new_state][action] == max_q:
            expected_q += q_table[new_state][action] * greedy_action_prob
          else:
            expected_q += q_table[new_state][action] * non_greedy_action_prob
        target = reward + discount_factor * expected_q

        next_action = self.chooseSarsaAction(q_table, state, epsilon)
        q_table[state, action] += learning_rate * (target - predict)
        state = new_state
        action = next_action

        rewards_current_episode += reward

      # Exponentially decay epsilon
      epsilon = min_epsilon + \
        (1 - min_epsilon) * np.exp(-epsilon_decay_rate*episode)
      rewards_all_episodes.append(rewards_current_episode)

    # Calculate and print the average reward per thousand episodes
    rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
    count = 1000

    print("\n********Average reward per thousand episodes********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r/1000)))
        count += 1000

    print("\n\n********Q-table********\n")
    print(q_table)

    # Plot change in deltas over iterations
    plt.figure(figsize=(20,10))
    # all_series = [list(x)[:50] for x in deltas.values()]
    for rewards in rewards_per_thousand_episodes:
      plt.plot(rewards)
    plt.show()

  def train_double_q_learning(self):
    learning_rate = 0.1
    epsilon = 1
    min_epsilon = 0.01
    epsilon_decay_rate = 0.001
    num_episodes = 10000
    rewards_all_episodes = []
    max_steps_per_episode = 100
    discount_rate = 0.99
    q_table_one = np.zeros((self.state_space_size, self.action_space_size))
    q_table_two = np.zeros((self.state_space_size, self.action_space_size))

    for episode in tqdm(range(num_episodes)):
      self.env.reset()
      done = False
      rewards_current_episode = 0

      for step in range(max_steps_per_episode):
        state = self.getState()

        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > epsilon:
          max_q_one = np.max(q_table_one[state,:])
          max_q_two = np.max(q_table_two[state,:])
          if max_q_one > max_q_two:
            action = np.argmax(q_table_one[state,:])
          else:
            action = np.argmax(q_table_two[state,:])
        else:
          action = self.getRandomAction()

        new_state, reward, done, info = self.env.step(action)
        new_state = self.getState()
        table_prob = random.uniform(0,1)
        if table_prob < 0.5:
          predict = q_table_one[state,action]
          next_max_action = np.argmax(q_table_one[new_state, :])
          q_table_one[state, action] += learning_rate * (reward +
            discount_rate * q_table_two[new_state, next_max_action] - predict)
        else:
          predict = q_table_two[state,action]
          next_max_action = np.argmax(q_table_two[new_state, :])
          q_table_two[state, action] += learning_rate * (reward +
            discount_rate * q_table_one[new_state, next_max_action] - predict)

        rewards_current_episode += reward

        if done:
          break

      # Exponentially decay epsilon
      epsilon = min_epsilon + \
        (1 - min_epsilon) * np.exp(-epsilon_decay_rate*episode)
      rewards_all_episodes.append(rewards_current_episode)

    # Calculate and print the average reward per thousand episodes
    rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
    count = 1000

    print("\n********Average reward per thousand episodes********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r/1000)))
        count += 1000

    print("\n\n********Q-table 1********\n")
    print(q_table_one)
    print("\n\n********Q-table 2********\n")
    print(q_table_two)

  def test_q_learning(self):
    self.test(q_learning_table)

  def test_td0(self):
    self.test(td_zero_table)

  def test_sarsa(self):
    self.test(sarsa_table)

  def test_expected_sarsa(self):
    self.test(expected_sarsa_table_two)

  def test_double_q_learning(self):
    self.test(double_q_learning_table_two)

  def test(self, q_table):
    self.env.reset()
    done = False
    max_steps_per_episode = 100

    for step in range(max_steps_per_episode):        
      self.env.render()
      time.sleep(0.3)
      state = self.getState()
      action = np.argmax(q_table[state,:])
      new_state, reward, done, info = self.env.step(action)

      if done:
        self.env.render()
        time.sleep(10)
        break

    self.env.close()

  def generateEpisode(self, V, discount_factor, learning_rate, deltas):
    # random_x = random.randint(1,3)
    # random_y = random.randint(1,3)
    # random_direction = random.randint(0,3)
    self.env.reset()
    pos = self.env.agent_pos
    direction = self.env.agent_dir
    state = (pos[0],pos[1],direction)
    steps = []
    max_steps_per_episode = 100

    for _ in range(max_steps_per_episode):
      if state[0] == 3 and state[1] == 3: # reached goal
        return

      action = self.getRandomAction()
      x, y, direction, reward = self.getNewStateAndRewardFromAction(state, action)
      new_state = (x,y,direction)
      steps.append([state,new_state])
      old_value = V[state]
      V[state] += learning_rate * (reward + discount_factor * V[new_state] - old_value)
      deltas[state].append(float(np.abs(old_value-V[state])))
      state = new_state
    return

  def chooseSarsaAction(self, q_table, state, epsilon):
    exploration_rate_threshold = random.uniform(0, 1)
    if exploration_rate_threshold > epsilon:
      action = np.argmax(q_table[state,:]) 
    else:
      action = self.getRandomAction()
    return action

  # Return random number between 0 and 2 (inclusive)
  def getRandomAction(self):
    return random.randint(0,2)

  def getNewStateAndRewardFromAction(self, state, action):
    self.env.reset()
    self.env.agent_pos = np.array((state[0], state[1]))
    self.env.agent_dir = state[2]
    observation, reward, done, info = self.env.step(action)
    x, y = self.env.agent_pos
    direction = self.env.agent_dir
    return (x, y, direction, reward)

  def getPolicyFromStateValueFunction(self,V):
    discount_factor = 0.99
    # Policy is represented as a Q-Table
    q_table = np.zeros((self.state_space_size, self.action_space_size))
    # Go through each state
    for i, state in enumerate(self.state_dict):
      # Try all the actions that could be taken from this state
      # and store the values of the resulting states in Q-Table
      for action in range(self.action_space_size):
        x, y, direction, reward = self.getNewStateAndRewardFromAction(state, action)
        new_state = (x,y,direction)
        new_state_value = V[new_state]
        q_table[i][action] = reward + discount_factor * new_state_value
    return q_table

  def getState(self):
    x,y = self.env.agent_pos
    direction = self.env.agent_dir
    return self.state_dict[(x,y,direction)]