from .FrozenLakeAgent import *

class MonteCarloAgent(FrozenLakeAgent):
  def train(self, env, num_episodes):
    q_table = np.zeros((self.state_space_size, self.action_space_size))
    # Holds the returns for every state across all episodes
    returns = {}
    # Holds the deltas for every state action pair across all episodes
    deltas = {}
    for state in range(self.state_space_size):
      for action in range(self.action_space_size):
        returns[(state,action)] = list()
        deltas[(state,action)] = list()
    discount_rate = 0.95

    for _ in tqdm(range(num_episodes)):
      episode = self.generateEpisode(env)
      G = 0
      for i, step in enumerate(episode[::-1]): # reverse the list
        # Every step is represented as an array with 4 elements:
        # initial state, action, reward, new_state
        G = discount_rate*G + step[2]
        initial_state = step[0]
        action = step[1]
        # If this step was not already visited in the episode
        if initial_state not in [x[0] for x in episode[::-1][len(episode)-i:]]:
          returns[(initial_state,action)].append(G)
          new_avg_state_action_value = np.average(returns[(initial_state,action)])
          delta = np.abs(q_table[initial_state][action] - new_avg_state_action_value)
          deltas[(initial_state, action)].append(delta)
          q_table[initial_state][action] = new_avg_state_action_value

    print("q_table", q_table)
    self.savePolicyFromQTable(q_table, 'mc_policy_1mil')

  def generateEpisode(self, env):
    state = env.reset()
    # 2-d array where each element is a step in the episode
    # represented as an array with 4 elements:
    # initial state, action, reward, new_state
    episode = []
    done = False
    while True:
      if done:
        return episode
      action = random.randint(0,3)
      new_state, reward, done, info = env.step(action)
      episode.append([state, action, reward, new_state])
      state = new_state
    return episode



