import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random

# parameters
gamma = 0.1 # discounting rate
rewardSize = -1
gridSize = 4
alpha = 0.1 # (0,1] // stepSize
terminationStates = [[0,0], [gridSize-1, gridSize-1]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
numIterations = 10000

# initialization
V = np.zeros((gridSize, gridSize))
returns = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
deltas = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]

# utils
def generateInitialState():
  initState = random.choice(states[1:-1])
  return initState

def generateNextAction():
  return random.choice(actions)

def takeAction(state, action):
  if list(state) in terminationStates:
    return 0, None
  finalState = np.array(state)+np.array(action)
  # if robot crosses wall
  if -1 in list(finalState) or gridSize in list(finalState):
    finalState = state
  return rewardSize, list(finalState)

for it in tqdm(range(numIterations)):
  state = generateInitialState()
  while True:
    action = generateNextAction()
    reward, finalState = takeAction(state, action)
    
    # we reached the end
    if finalState is None:
      break
    
    # modify Value function
    before =  V[state[0], state[1]]
    V[state[0], state[1]] += alpha*(reward + gamma*V[finalState[0], finalState[1]] - V[state[0], state[1]])
    deltas[state[0], state[1]].append(float(np.abs(before-V[state[0], state[1]])))
    
    state = finalState

plt.figure(figsize=(20,10))
all_series = [list(x)[:50] for x in deltas.values()]

for state, state_deltas in deltas.items():
  print(f"State: {state}, Deltas", state_deltas[:10])

# for series in all_series:
#   plt.plot(series)

# plt.show()