# myAgentP3.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
# This file was based on the starter code for student bots, and refined 
# by Mesut (Xiaocheng) Yang


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
from copy import copy
from math import inf
from distanceCalculator import Distancer

def ExpectedMax(gameState, action, depth, depthLimit, agentIndex, recursionLevel, doLogging, evalFunction, broadcast):
  # Log starting info
  indent = ""
  for _ in range(0, recursionLevel):
    indent = indent + "  "
  if doLogging:
    print("{0}Max(agent={1}, previousAction={2}, depth={3}, depthLimit={4})".format(indent, agentIndex, action, depth, depthLimit))

  # Check for terminal state
  if gameState.isOver() or depth == depthLimit:
    if doLogging:
      print("{0}Terminal State, returning (action={1}, score={2})".format(indent + "  ", action, evalFunction(gameState, broadcast)))
    return evalFunction(gameState, broadcast)

  # Calculate max action recursively
  else:
    maxValue = -inf
    nextAgent = (agentIndex + 1) % gameState.getNumAgents()

    if agentIndex == 0:
      legalActions = [broadcast[0]]
    else:
      legalActions = gameState.getLegalActions(agentIndex)
      legalActions = actionsWithoutReverse(actionsWithoutStop(legalActions), gameState, 1)

    for nextAction in legalActions:
      # print("Generating successor for agent", agentIndex)
      nextState = gameState.generateSuccessor(agentIndex, nextAction)
      if nextAgent != 0 and nextAgent != 1:
        result = ExpectedMin(nextState, nextAction, depth, depthLimit, nextAgent, recursionLevel + 1, doLogging, evalFunction, broadcast)
      else:
        result = ExpectedMax(nextState, nextAction, depth, depthLimit, nextAgent, recursionLevel + 1, doLogging, evalFunction, broadcast)
      if result > maxValue:
        maxValue = result

    # Log ending info
    if doLogging:
        print("{0}Max returning cost={1} for action={2}".format(indent, maxValue, action))
    return maxValue

def ExpectedMin(gameState, action, depth, depthLimit, agentIndex, recursionLevel, doLogging, evalFunction, broadcast):

  # Log starting info
  indent = ""
  for _ in range(0, recursionLevel):
    indent = indent + "  "
  if doLogging:
    print("{0}Min(agent={1}, previousAction={2}, depth={3}, depthLimit={4})".format(indent, agentIndex, action, depth, depthLimit))

  # Get next agent
  nextAgent = (agentIndex + 1) % gameState.getNumAgents()

  # Check for terminal state
  if gameState.isOver() or depth == depthLimit:
    if doLogging:
      print("{0}Terminal State, returning (action={1}, score={2}".format(indent + "  ", action, evalFunction(gameState, broadcast)))
    return evalFunction(gameState, broadcast)

  # Calculate min action recursively
  else:
    # totalScore = 0
    legalActions = gameState.getLegalActions(agentIndex)

    scores = []
    for nextAction in legalActions:
      nextState = gameState.generateSuccessor(agentIndex, nextAction)
      if nextAgent == 0 or nextAgent == 1:
        scores.append(ExpectedMax(nextState, nextAction, depth + 1, depthLimit, nextAgent, recursionLevel + 1, doLogging, evalFunction, broadcast))
      else:
        scores.append(ExpectedMin(nextState, nextAction, depth, depthLimit, nextAgent, recursionLevel + 1, doLogging, evalFunction, broadcast))

    minScore = min(scores)
    lowScores = [score for score in scores if score == minScore]
    lowScoreChance = 0.75 / len(lowScores)
    highScoreChance = 0.25 / (len(scores) - len(lowScores))

    score = 0
    for value in scores:
      if value == minScore:
        score += value * lowScoreChance
      else:
        score += value * highScoreChance
    # score = totalScore / len(legalActions)

    # Log ending info
    if doLogging:
      print("{0}Min returning cost={1})".format(indent, score))
    return score

def Expectimax(gameState, agentIndex, depth, depthLimit, doLogging, evalFunction, broadcast):
  maxValue = -inf
  bestAction = None
  # actionsWithoutReverse(actionsWithoutStop(gameState.getLegalActions(agentIndex)), gameState, agentIndex):
  for action in actionsWithoutStop(gameState.getLegalActions(agentIndex)):
    nextState = gameState.generateSuccessor(agentIndex, action)
    result = ExpectedMin(nextState, action, depth, depthLimit, 2, 0, doLogging, evalFunction, broadcast)
    if result > maxValue:
      maxValue = result
      bestAction = copy(action)
  if doLogging:
    print(bestAction)
  return bestAction

#########
# Agent #
#########
class MyAgent(CaptureAgent):
  """
  YOUR DESCRIPTION HERE
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    # Make sure you do not delete the following line. 
    # If you would like to use Manhattan distances instead 
    # of maze distances in order to save on initialization 
    # time, please take a look at:
    # CaptureAgent.registerInitialState in captureAgents.py.
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)

    self.depth = 0
    self.screenHeight = 18
    self.screenWidth = 34
    self.initialFoodCount = gameState.getFood().count()

    self.distancer = Distancer(gameState.data.layout)

    # file = open("weights.dat", "r")
    # self.featureWeights = []
    # for line in file:
    #   self.featureWeights.append(int(line))
    # file.close()

    self.featureWeights = [
      -5,                     # 1. Distance to food not being targeted by teammate
      -1200,                  # 2. Number of ghosts 0 blocks away
      -615,                   # 3. Number of ghosts 1 blocks away
      -50,                    # 4. Number of ghosts 2 blocks away
      0,                      # 5. Number of ghosts 3 blocks away
      612                     # 6. Food eaten
    ]

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    teammateActions = self.receivedBroadcast
    action = Expectimax(gameState, 1, 0, self.depth, False, self.evaluationFunction, teammateActions)
    return action

  def evaluationFunction(self, gameState, broadcast):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # start = time.time()
    # Features
    # 1. Distance to food not being targeted by teammate
    # 2. Number of ghosts 3 blocks away
    # 3. Number of ghosts 2 blocks away
    # 4. Number of ghosts 1 blocks away
    # 5. Number of ghosts 0 blocks away
    # 6. Food eaten

    # Positions
    teammatePosition = gameState.getAgentPosition(0)
    pacmanPosition = gameState.getAgentPosition(1)
    ghostPositions = []
    for ghostIndex in gameState.getGhostTeamIndices():
      ghostPositions.append(gameState.getAgentPosition(ghostIndex))

    # 
    # 1. Distance to food not being targeted by teammate (minFoodDistance)
    # 

    # Figure out which pellet the staff agent is targeting
    food = gameState.getFood().asList()
    teammateFoodTarget = None

    if len(food):
      minTeammateFoodDistance = inf
      for pellet in food:
        distance = self.distancer.getDistance(teammatePosition, pellet)
        if distance < minTeammateFoodDistance:
          minTeammateFoodDistance = distance
          teammateFoodTarget = pellet

    # Get min distance to other food pellet
    if len(food):
      minFoodDistance = inf
      for pellet in food:
        if pellet == teammateFoodTarget and len(food) > 2:
          continue
        distance = GetAStarDist(pacmanPosition, ghostPositions, pellet, gameState.getWalls())
        minFoodDistance = min(minFoodDistance, distance)
    else:
      minFoodDistance = 0

    #
    # 2-5. Number of ghosts 0,1,2,3 blocks away (nearbyGhosts)
    #
    nearbyGhosts = [0, 0, 0, 0]
    for ghostPosition in ghostPositions:
      distance = self.distancer.getDistance(pacmanPosition, ghostPosition)
      try:
        nearbyGhosts[distance] += 1
      except IndexError:
        continue

    #
    # 6. Food Eaten (score)
    #
    score = gameState.getScore()

    # Calculate utility
    features = [minFoodDistance]
    features.extend(nearbyGhosts)
    features.extend([score])
    utility = 0
    for index, feature in enumerate(features):
      utility += feature * self.featureWeights[index]

    return utility

def actionsWithoutStop(legalActions):
  """
  Filters actions by removing the STOP action
  """
  legalActions = list(legalActions)
  if Directions.STOP in legalActions:
    legalActions.remove(Directions.STOP)
  return legalActions

def actionsWithoutReverse(legalActions, gameState, agentIndex):
  """
  Filters actions by removing REVERSE, i.e. the opposite action to the previous one
  """
  legalActions = list(legalActions)
  reverse = Directions.REVERSE[gameState.getAgentState(agentIndex).configuration.direction]
  if len (legalActions) > 1 and reverse in legalActions:
    legalActions.remove(reverse)
  return legalActions

def GetAStarDist(position, ghostPositions, destination, wallGrid):
  wallGridTemp = wallGrid.copy()
  for (x, y) in ghostPositions:
    wallGridTemp[x][y] = True

  possibleMoves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  fringe = util.PriorityQueue()
  exploredNodes = [position]
  x,y = position

  if position == destination:
    return 0

  for dx,dy in possibleMoves:
    if not wallGridTemp[x + dx][y + dy]:
      newPosition = (x + dx, y + dy)
      if newPosition != destination:
        fringe.push([position, newPosition], 1 + util.manhattanDistance(newPosition, destination))
      else:
        return 1

  while not fringe.isEmpty():
    path = fringe.pop()
    node = path[-1]

    if node == destination:
      return len(path) - 1
    
    if node not in exploredNodes:
      exploredNodes.append(node)
      nodeX,nodeY = node
      for dx,dy in possibleMoves:
        if not wallGridTemp[nodeX + dx][nodeY + dy] and (nodeX + dx, nodeY + dy) not in exploredNodes:
          newPath = path.copy()
          newPath.append((nodeX + dx, nodeY + dy))
          fringe.push(newPath, len(path) -1 + util.manhattanDistance((nodeX + dx, nodeY + dy), destination))

  return 99999999
