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
import numpy as np
import cv2 as cv
import os

#########
# Agent #
#########
class MyAgent(CaptureAgent):
  """
  YOUR DESCRIPTION HERE
  """
  def __init__(self, index, timeForComputing = .1):
    super(MyAgent, self).__init__(index, timeForComputing)
    self.previousState = None
    self.previousScore = 0
    self.previousAction = None

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

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    print(self.previousState)
    height = gameState.getWalls().height
    width = gameState.getWalls().width
    image = np.zeros((width, height, 3), dtype=np.uint8)
    
    # Add walls
    gray = (145, 145, 145)
    for (x, y) in gameState.getWalls().asList():
      image[x, y] = gray

    yellow = (0, 200, 255)
    for (x, y) in gameState.getFood().asList():
      image[x, y] = yellow

    green = (0, 255, 0)
    for index in gameState.getPacmanTeamIndices():
      position = gameState.getAgentPosition(index)
      if position is not None:
        (x, y) = position
        image[x, y] = green

    red = (0, 0, 255)
    for index in gameState.getGhostTeamIndices():
      position = gameState.getAgentPosition(index)
      if position is not None:
        (x, y) = position
        image[x, y] = red

    # Save state
    trainingDir = "training"
    if not os.path.exists(trainingDir):
      os.mkdir(trainingDir)
    
    if self.previousState is not None:
      decisionLog = open(os.path.join(trainingDir, "decisionLog.txt"), 'a')

      timeString = str(time.time())
      scoreDiff = gameState.getScore() - self.previousScore
      scoreDiff -= 1
      decisionLog.write(timeString + "," + str(self.previousAction) + "," + str(scoreDiff) + "\n")
      cv.imwrite(os.path.join(trainingDir, "frame-" + timeString + ".png"), image)
      decisionLog.close()

    # cv.imshow("demo", image)
    # cv.waitKey(0)

    teammateActions = self.receivedBroadcast
    # Process your teammate's broadcast! 
    # Use it to pick a better action for yourself

    actions = gameState.getLegalActions(self.index)

    filteredActions = actionsWithoutReverse(actionsWithoutStop(actions), gameState, self.index)


    chosenAction = random.choice(actions) # Change this!
    self.previousAction = chosenAction
    self.previousState = image
    self.previousScore = gameState.getScore()
    return chosenAction

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
