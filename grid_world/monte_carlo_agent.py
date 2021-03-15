import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
# Optimal Q Table found after training
from q_tables import monte_carlo_table as q_table
q_table = np.array(q_table)

class MonteCarloAgent:
  def __init__(self, env, state_dict, state_space_size, action_space_size):
    self.env = env
    self.state_dict = state_dict
    self.state_space_size = state_space_size
    self.action_space_size = action_space_size

  def train(self):
    num_episodes = 10000
    discount_rate = 0.99
    # Holds the returns for every state across all episodes
    returns = {key:list() for key in self.state_dict.keys()}
    # Holds the deltas in state value for every state across all episodes
    deltas = {key:list() for key in self.state_dict.keys()}
    # Holds the state values (initially 0 for all states)
    V = {key:0 for key in self.state_dict.keys()}

    for _ in tqdm(range(num_episodes)):
      episode = self.generateEpisode()
      G = 0
      for i, step in enumerate(episode[::-1]): # reverse the list
        # Every step is represented as an array with 4 elements:
        # initial state, action, reward, new_state
        G = discount_rate*G + step[2]
        initial_state = step[0]
        # If this step was not already visited in the episode
        if initial_state not in [x[0] for x in episode[::-1][len(episode)-i:]]:
          returns[initial_state].append(G)
          new_avg_state_value = np.average(returns[initial_state])
          delta = np.abs(V[initial_state] - new_avg_state_value)
          deltas[initial_state].append(delta)
          V[initial_state] = new_avg_state_value

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

  def test(self):
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

  def generateEpisode(self):
    random_x = random.randint(1,3)
    random_y = random.randint(1,3)
    random_direction = random.randint(0,3)
    state = (random_x,random_y,random_direction)
    # 2-d array where each element is a step in the episode
    # represented as an array with 4 elements:
    # initial state, action, reward, new_state
    episode = []

    while True:
      if state[0] == 3 and state[1] == 3: # reached goal
        return episode
      action = self.getRandomAction()
      x, y, direction, reward = self.getNewStateAndRewardFromAction(state, action)
      new_state = (x,y,direction)
      episode.append([state, action, reward, new_state])
      state = new_state
    return episode

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
    discount_rate = 0.99
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
        q_table[i][action] = reward + discount_rate * new_state_value
    return q_table

  def getState(self):
    x,y = self.env.agent_pos
    direction = self.env.agent_dir
    return self.state_dict[(x,y,direction)]