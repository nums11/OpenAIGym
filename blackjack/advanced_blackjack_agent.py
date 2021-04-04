import numpy as np
from tqdm import tqdm
import random

"""
Adding the dealers hand to the state space
State-space size: 380
"""
class AdvancedBlackjackAgent:
	def train(self, env, num_episodes):
		state_space_size = 2
		action_space_size = 2
		learning_rate = 0.1
		epsilon = 1
		min_epsilon = 0.1
		epsilon_decay_rate = 0.001
		rewards_all_episodes = []
		# discount_rate = 0.99
		discount_rate = 1
		q_table = {}
		bool = False
		for player_sum in range(2,21):
			for dealer_sum in range(1, 11):
				q_table[(player_sum, dealer_sum, bool)] = np.zeros((action_space_size))
				bool = not bool
				q_table[(player_sum, dealer_sum, bool)] = np.zeros((action_space_size))
				bool = not bool

		for episode in tqdm(range(num_episodes)):
		  player_sum, dealer_sum, has_ace = env.reset()
		  if player_sum == 21:
		  	rewards_all_episodes.append(1)
		  	continue
		  state = (player_sum, dealer_sum, has_ace)
		  rewards_current_episode = 0

		  while True:
		    exploration_rate_threshold = random.uniform(0, 1)
		    if exploration_rate_threshold > epsilon:
		      action = np.argmax(q_table[state]) 
		    else:
		      action = random.randint(0,1)

		    observation, reward, done, info = env.step(action)
		    player_sum, dealer_sum, has_ace = observation

		    new_state = (player_sum, dealer_sum, has_ace)
		    new_state_value = 0
		    # Make sure the state is not terminal
		    if new_state in q_table:
		    	new_state_value = np.max(q_table[new_state])

		    # Update Q-table for Q(s,a)
		    q_table[state][action] = q_table[state][action] * (1 - learning_rate) + \
		      learning_rate * (reward + discount_rate * new_state_value)

		    rewards_current_episode += reward

		    if done:
		      break

		  # Exponentially decay epsilon
		  epsilon = min_epsilon + \
		    (1 - min_epsilon) * np.exp(-epsilon_decay_rate*episode)
		  rewards_all_episodes.append(rewards_current_episode)

		self.printRewards(num_episodes, rewards_all_episodes)

	def printRewards(self, num_episodes, rewards):
		split = num_episodes / 10
		rewards_per_split_episodes = np.split(np.array(rewards),num_episodes/split)
		count = split

		print(f"\n********Average reward per {split} episodes ********\n")
		for r in rewards_per_split_episodes:
		  print(count, ": ", str(sum(r/split)))
		  count += split


