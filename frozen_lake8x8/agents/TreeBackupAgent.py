from .FrozenLakeAgent import *

class TreeBackupAgent(FrozenLakeAgent):
  def train(self, env, n, num_episodes):
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
    q_table = np.zeros((self.state_space_size, self.action_space_size))
    farthest_states = {}

    for episode in tqdm(range(num_episodes)):
      state = env.reset()
      rewards_current_episode = 0
      farthest_state_for_episode = 0
      t = 0
      T = np.inf

      action = None
      exploration_rate_threshold = random.uniform(0, 1)
      if exploration_rate_threshold > epsilon:
        action = np.argmax(q_table[state]) 
      else:
        action = random.randint(0,3)

      actions = [action]
      states = [0]
      rewards = [0]

      for step in range(max_steps_per_episode):
        if t < T:
          if state > farthest_state_for_episode:
            farthest_state_for_episode = state

          new_state, reward, done, info = env.step(action)
          states.append(new_state)
          rewards.append(reward)

          if done:
            T = t + 1
            if reward > 0:
              print("I won!", reward)
          else:
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > epsilon:
              action = np.argmax(q_table[state]) 
            else:
              action = random.randint(0,3)
            actions.append(action)

        tau = t - n + 1
        if tau >= 0:
          if t + 1 >= T:
            G = rewards[T]
          else:
            expectation = 0
            max_action = np.argmax(q_table[states[t+1]])
            for possible_action in range(self.action_space_size):
              prob = 0
              if possible_action == max_action:
                prob = 1 - epsilon
              else:
                prob = epsilon
              expectation += prob * q_table[states[t+1]][possible_action]
            G = rewards[t+1] + discount_rate * expectation

          for k in reversed(range(tau + 1, min(t, T-1))):
            A_k = actions[k]
            max_action_at_k = np.argmax(q_table[states[k]])
            A_k_prob = None
            if A_k == max_action_at_k:
              A_k_prob = 1 - epsilon
            else:
              A_k_prob = epsilon

            expectation = 0
            for possible_action in range(self.action_space_size):
              if possible_action == A_k:
                continue
              prob = 0
              if possible_action == max_action_at_k:
                prob = 1 - epsilon
              else:
                prob = epsilon
              expectation += prob * q_table[states[k]][possible_action]

            G = rewards[k] + discount_rate * expectation + discount_rate * A_k_prob * G

          q_table[states[tau]][actions[tau]] = q_table[states[tau]][actions[tau]] + \
            learning_rate * (G - q_table[states[tau]][actions[tau]])

        state = new_state
        rewards_current_episode += reward

        if tau == T - 1:
          break

        t += 1

      # Exponentially decay epsilon
      epsilon = min_epsilon + \
        (1 - min_epsilon) * np.exp(-epsilon_decay_rate*episode)
      rewards_all_episodes.append(rewards_current_episode)

      if farthest_state_for_episode in farthest_states:
        farthest_states[farthest_state_for_episode] += 1
      else:
        farthest_states[farthest_state_for_episode] = 0

    self.printRewards(num_episodes/10, num_episodes, rewards_all_episodes)
    print("q_table", q_table)
    self.savePolicyFromQTable(q_table, str(n) + '_step_tree_backup_policy_100k')
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



