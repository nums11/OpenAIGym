from .FrozenLakeAgent import *

class DoubleQLearningAgent(FrozenLakeAgent):
  def train(self, env, num_episodes):
    # learning_rate = 0.1
    # learning_rate = 0.05
    learning_rate = 0.01
    epsilon = 1
    min_epsilon = 0.1
    epsilon_decay_rate = 0.0001
    rewards_all_episodes = []
    discount_rate = 0.99
    # discount_rate = 0.9
    # discount_rate = 1
    max_steps_per_episode = 300
    q_table_one = np.zeros((self.state_space_size, self.action_space_size))
    q_table_two = np.zeros((self.state_space_size, self.action_space_size))
    farthest_states = {}

    for episode in tqdm(range(num_episodes)):
      state = env.reset()
      rewards_current_episode = 0
      farthest_state_for_episode = 0

      for step in range(max_steps_per_episode):
        if state > farthest_state_for_episode:
          farthest_state_for_episode = state

        is_done = False
        done_reward = 0
        use_q_one_prob = random.uniform(0,1)
        if use_q_one_prob < 0.5:
          exploration_rate_threshold = random.uniform(0, 1)
          if exploration_rate_threshold > epsilon:
            action = np.argmax(q_table_one[state]) 
          else:
            action = random.randint(0,3)

          new_state, reward, done, info = env.step(action)
          q_table_one[state][action] = q_table_one[state][action] + learning_rate * (
            reward + discount_rate * np.max(q_table_two[new_state]) - q_table_one[state][action])
          state = new_state
          rewards_current_episode += reward

          if done:
            is_done = True
            done_reward = reward

        else:
          exploration_rate_threshold = random.uniform(0, 1)
          if exploration_rate_threshold > epsilon:
            action = np.argmax(q_table_two[state]) 
          else:
            action = random.randint(0,3)

          new_state, reward, done, info = env.step(action)
          q_table_two[state][action] = q_table_two[state][action] + learning_rate * (
            reward + discount_rate * np.max(q_table_one[new_state]) - q_table_two[state][action])
          state = new_state
          rewards_current_episode += reward

          if done:
            is_done = True
            done_reward = reward

        if is_done:
          # print("Reached goal or hole", reward)
          if done_reward > 0:
            print("I won!", reward)
          break

      # Exponentially decay epsilon
      epsilon = min_epsilon + \
        (1 - min_epsilon) * np.exp(-epsilon_decay_rate*episode)
      rewards_all_episodes.append(rewards_current_episode)

      if farthest_state_for_episode in farthest_states:
        farthest_states[farthest_state_for_episode] += 1
      else:
        farthest_states[farthest_state_for_episode] = 0

    self.printRewards(num_episodes/10, num_episodes, rewards_all_episodes)
    print("q_table_one", q_table_one)
    print("q_table_two", q_table_two)
    self.saveDQLPolicyFromTables(q_table_one, q_table_two, 'dql_policy_100k')
    farthest_states_ordered = collections.OrderedDict(sorted(farthest_states.items()))
    self.plotFarthestStates(farthest_states_ordered)

  def plotFarthestStates(self, farthest_states_ordered):
    print("farthest_states_ordered", farthest_states_ordered)
    states = list(farthest_states_ordered.keys())
    frequencies = list(farthest_states_ordered.values())
    plt.bar(states,frequencies, width = 0.4)
    plt.xlabel("States")
    plt.ylabel("Frequencies")
    plt.title("Frequency of farthest states")
    plt.show()

  def saveDQLPolicyFromTables(self, q_table_one, q_table_two, policy_name):
    policy = {}
    for state in range(self.state_space_size):
      max_action_1 = np.argmax(q_table_one[state])
      max_action_2 = np.argmax(q_table_two[state])
      max_action = None
      if max_action_1 != max_action_2:
        use_q_one_prob = random.uniform(0,1)
        if use_q_one_prob < 0.5:
          max_action = max_action_1
        else:
          max_action = max_action_2
      else:
        max_action = max_action_1
      policy[state] = max_action
    print("policy", policy)
    np.save('policies/' + policy_name + '.npy', policy)





