
def getPositionAndDirectionFromState(state):
  state_position_dict = {
    0: (1,1,0),
    1: (1,1,1),
    2: (1,1,2),
    3: (1,1,3),
    4: (2,1,0),
    5: (2,1,1),
    6: (2,1,2),
    7: (2,1,3),
    8: (3,1,0),
    9: (3,1,1),
    10: (3,1,2),
    11: (3,1,3),
    12: (1,2,0),
    13: (1,2,1),
    14: (1,2,2),
    15: (1,2,3),
    16: (2,2,0),
    17: (2,2,1),
    18: (2,2,2),
    19: (2,2,3),
    20: (3,2,0),
    21: (3,2,1),
    22: (3,2,2),
    23: (3,2,3),
    24: (1,3,0),
    25: (1,3,1),
    26: (1,3,2),
    27: (1,3,3),
    28: (2,3,0),
    29: (2,3,1),
    30: (2,3,2),
    31: (2,3,3),
    32: (3,3,0),
    33: (3,3,1),
    34: (3,3,2),
    35: (3,3,3),
  }
  return state_position_dict[state]

def policy_evaluation(policy):
  max_iterations = 10000
  theta = .0000001
  # Number of evaluation_iterations
  evaluation_iterations = 1
  # Initialize a value function for each state as zero
  V = np.zeros(state_space_size)
  # Repeat until change in value is below threshold or max iterations
  for i in range(max_iterations):
    # Init a change of value function as 0
    delta = 0
    for state in range(state_space_size):
      # New value for current state
      v = 0
      # Try all possible actions that could be taken from this state
      for action in range(len(policy[state])):
        action_prob = policy[state][action]
        # Check how good next state will be from taking the current action
        x,y,direction = getPositionAndDirectionFromState(state)
        env.agent_pos = (x,y)
        env.agent_dir = direction
        observation, reward, done, info = env.step(action)
        next_state = getState()
        v += action_prob * 1 * (reward + discount_rate) * V[next_state]

      # Calculate the change in value
      delta = max(delta, np.abs(V[state] - v))
      # Update state value function
      V[state] = v
    evaluation_iterations += 1

    # Terminate if value change is insignificant (convergence)
    if delta < theta:
      print(f'State value function found in {evaluation_iterations} iterations')
      return V

def one_step_lookahead(state, V):
  action_values = np.zeros(action_space_size)
  for action in range(action_space_size):
    x,y,direction = getPositionAndDirectionFromState(state)
    env.agent_pos = (x,y)
    env.agent_dir = direction
    observation, reward, done, info = env.step(action)
    next_state = getState()
    action_values[action] += reward * discount_rate * V[next_state]
  return action_values

# FDP
def train_policy_iteration():
  max_iterations = 10000
  # Start with a random policy
  policy = np.zeros([state_space_size, action_space_size])
  evaluated_policies = 1
  # Repeat until convergence or max num iterations
  for i in range(max_iterations):
    stable_policy = True
    # Evaluate current policy
    V = policy_evaluation(policy)
    # Go through Each state and try to improve the actions that were
    # taken (policy improvement)
    for state in range(state_space_size):
      # Choose the best action for the current state under the policy
      current_action = np.argmax(policy[state])
      # Look one step ahead and evaluate if current action is optimal
      # (trying every possible action from current state)
      action_values = one_step_lookahead(state,V)
      print(action_values)
      best_action = np.argmax(action_values)
      # If action didn't change
      if current_action != best_action:
        stable_policy = True
        policy[state] = np.eye(action_space_size)[best_action]
        print("policy[state]", policy[state])

    evaluated_policies += 1
    # If algorithm converged
    if stable_policy:
      print(f'Evaluated {evaluated_policies} policies.')
      print("optimal_policy", policy)
      print("optimal state value function", V)
      return policy, V