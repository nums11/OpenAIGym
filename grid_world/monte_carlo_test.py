import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random

# parameters
gamma = 0.6 # discounting rate
rewardSize = -1
gridSize = 4
terminationStates = [[0,0], [gridSize-1, gridSize-1]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
numIterations = 10000

# initialization
V = np.zeros((gridSize, gridSize))
returns = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
deltas = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]

# utils
def generateEpisode():
  initState = random.choice(states[1:-1])
  episode = []
  while True:
    if list(initState) in terminationStates:
      return episode
    action = random.choice(actions)
    finalState = np.array(initState)+np.array(action)
    if -1 in list(finalState) or gridSize in list(finalState):
      finalState = initState
    episode.append([list(initState), action, rewardSize, list(finalState)])
    initState = finalState

for it in tqdm(range(numIterations)):
  episode = generateEpisode()
  G = 0
  for i, step in enumerate(episode[::-1]): # reversed the list
    G = gamma*G + step[2]
    # If this step was not already visited in the episode
    if step[0] not in [x[0] for x in episode[::-1][len(episode)-i:]]:
      idx = (step[0][0], step[0][1])
      returns[idx].append(G)
      newValue = np.average(returns[idx])
      deltas[idx[0], idx[1]].append(np.abs(V[idx[0], idx[1]]-newValue))
      V[idx[0], idx[1]] = newValue

print(V)

# plt.figure(figsize=(20,10))
# all_series = [list(x)[:50] for x in deltas.values()]
# for series in all_series:
#   plt.plot(series)

# plt.show()



