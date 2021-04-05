import numpy as np
from tqdm import tqdm
import random

"""
State-space is just the player current sum and
whether or not they have an ace
State-space size: 42
"""
class SimpleQLearningAgent:
	def train(self, env, num_episodes):
		state_space_size = 2
		action_space_size = 2
		# learning_rate = 0.1
		# learning_rate = 0.05
		learning_rate = 0.01
		epsilon = 1
		min_epsilon = 0.1
		epsilon_decay_rate = 0.001
		rewards_all_episodes = []
		# discount_rate = 0.99
		discount_rate = 1
		q_table = {}
		bool = False
		for i in range(1,22):
			q_table[(i,bool)] = np.zeros((action_space_size))
			bool = not bool
			q_table[(i,bool)] = np.zeros((action_space_size))
			bool = not bool

		for episode in tqdm(range(num_episodes)):
		  player_sum, dealer_sum, has_ace = env.reset()
		  if player_sum == 21:
		  	rewards_all_episodes.append(1)
		  	continue
		  state = (player_sum, has_ace)
		  rewards_current_episode = 0

		  while True:
		    exploration_rate_threshold = random.uniform(0, 1)
		    if exploration_rate_threshold > epsilon:
		      action = np.argmax(q_table[state]) 
		    else:
		      action = random.randint(0,1)

		    observation, reward, done, info = env.step(action)
		    player_sum, dealer_sum, has_ace = observation

		    new_state = (player_sum, has_ace)
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
		print("q_table", q_table)
		policy = self.getPolicyFromQTable(q_table)
		self.savePolicy(policy)
		# self.saveRewards(rewards_all_episodes)

	def printRewards(self, num_episodes, rewards):
		split = 100000	# 100k
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
	  np.save('simple_ql_policy.npy', policy)

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

	    state = (player_sum, has_ace)
	    done = False
	    while True:
	      if done:
	        rewards_all_episodes.append(reward)
	        break
	      action = policy[state]
	      observation, reward, done, info = env.step(action)
	      player_sum, dealer_sum, has_ace = observation
	      new_state = (player_sum,has_ace)
	      state = new_state

	  avg_rewards = sum(rewards_all_episodes) / num_episodes
	  print(f'******* Average rewards across {num_episodes} episodes *******')
	  print(avg_rewards)



