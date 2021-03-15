import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
# Optimal Q Table found after training
from q_tables import value_iteration_table_three as q_table
q_table = np.array(q_table)
print(q_table)

class ValueIterationAgent:
  def __init__(self, env, state_dict, state_space_size, action_space_size):
    self.env = env
    self.state_dict = state_dict
    self.state_space_size = state_space_size
    self.action_space_size = action_space_size


  def train(self):
    discount_rate = 0.99
    # Default states with random values between 0 and 1
    V = {key: random.uniform(0,1) for key in self.state_dict.keys()}
    # Set terminal state to have a value of 0
    for i in range(4):
      V[(3,3,i)] = 0
    biggest_deltas = []

    # Repeat until convergence
    num_iterations = 0
    for _ in tqdm(range(10000)):
      biggest_delta = 0
      threshold = 1e-4

      for state in self.state_dict:
        old_state_value = V[state]

        # Not terminal state
        if state[0] != 3 and state[1] != 3:
          new_state_value = float('-inf')

          for action in range(self.action_space_size):
            x, y, direction, reward = self.getNewStateAndRewardFromAction(state, action)
            new_state = (x,y,direction)
            v = reward + discount_rate * V[new_state]
            if v > new_state_value:
              new_state_value = v
          V[state] = new_state_value
          biggest_delta = max(biggest_delta, np.abs(old_state_value - V[state]))
          biggest_deltas.append(biggest_delta)

      num_iterations += 1
      # if biggest_delta < threshold:
      #   break

    print(f'Converged after {num_iterations} iterations')
    print('State Value function', V)
    policy = self.getPolicyFromStateValueFunction(V)
    print('Policy', policy)
    plt.figure(figsize=(20,10))
    biggest_deltas = biggest_deltas[:100]
    plt.plot(biggest_deltas)
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