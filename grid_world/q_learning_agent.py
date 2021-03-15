import numpy as np
import random
import time
from tqdm import tqdm
# Optimal Q Table found after training
from q_table import q_table
q_table = np.array(q_table)

class QLearningAgent:
  def __init__(self, env, state_dict, state_space_size, action_space_size):
    self.env = env
    self.state_dict = state_dict
    self.q_table = np.zeros((state_space_size, action_space_size))

  def train(self):
    print('Training Q-Learning Agent')
    learning_rate = 0.1
    epsilon = 1
    min_epsilon = 0.01
    epsilon_decay_rate = 0.001
    num_episodes = 10000
    rewards_all_episodes = []
    max_steps_per_episode = 100
    discount_rate = 0.99

    for episode in tqdm(range(num_episodes)):
      self.env.reset()
      done = False
      rewards_current_episode = 0

      for step in range(max_steps_per_episode):
        state = self.getState()

        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > epsilon:
          action = np.argmax(self.q_table[state,:]) 
        else:
          action = self.getRandomAction()

        new_state, reward, done, info = self.env.step(action)
        new_state = self.getState()
        # Update Q-table for Q(s,a)
        self.q_table[state, action] = self.q_table[state, action] * (1 - learning_rate) + \
          learning_rate * (reward + discount_rate * np.max(self.q_table[new_state, :]))

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
    print(self.q_table)

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

  def getState(self):
    x,y = self.env.agent_pos
    direction = self.env.agent_dir
    return self.state_dict[(x,y,direction)]

  # Return random number between 0 and 2 (inclusive)
  def getRandomAction(self):
    return random.randint(0,2)