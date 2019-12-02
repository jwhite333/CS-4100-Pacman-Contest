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
import math
from math import inf
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# resize = T.Compose([T.ToPILImage(),
#                     T.Resize(40, interpolation=Image.CUBIC),
#                     T.ToTensor()])

###########
# PyTorch #
###########
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

# class ReplayMemory(object):

#     def __init__(self):
#         self.trainingDir = "training"
#         if not os.path.exists(self.trainingDir):
#           os.mkdir(self.trainingDir)
#         self.file = os.path.join(self.trainingDir, "memory.dat")

#     def push(self, state, action, next_state, reward):
#         """Saves a transition."""

#         # Save image
#         timeString = str(time.time())
#         cv.imwrite(os.path.join(self.trainingDir, "state-" + timeString + ".png"), state)
#         cv.imwrite(os.path.join(self.trainingDir, "next_state-" + timeString + ".png"), next_state)

#         # Save the rest
#         file = open(self.file, 'a')
#         file.write("state-" + timeString + ".png," +
#                    str(action) + "," +
#                    "next_state-" + timeString + ".png," +
#                    str(reward + "\n"))


#         file.close()

#     def sample(self, batch_size):
#         file = open(self.file, 'r')
#         lines = random.sample(file, batch_size)
#         transitions = []
#         for line in lines:
#           line = line.split(",")
#           state = cv.imread(os.path.join(self.trainingDir, "state-" + line[0] + ".png"))
#           next_state = cv.imread(os.path.join(self.trainingDir, "next_state-" + line[2] + ".png"))
#           transitions.append(Transition(state, line[1], next_state, float(line[3])))
#         file.close()
#         return transitions

#     def __len__(self):
#         length = 0
#         file = open(self.file, 'r')
#         length = len(file)
#         file.close()
#         return length

# class DQN(nn.Module):

#     def __init__(self, h, w, outputs):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
#         self.bn3 = nn.BatchNorm2d(32)

#         # Number of Linear input connections depends on output of conv2d layers
#         # and therefore the input image size, so compute it.
#         def conv2d_size_out(size, kernel_size = 5, stride = 1):
#             return (size - (kernel_size - 1) - 1) // stride  + 1
#         convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
#         convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
#         linear_input_size = convw * convh * 32
#         self.head = nn.Linear(linear_input_size, outputs)
#     # Called with either one element to determine next action, or a batch
#     # during optimization. Returns tensor([[left0exp,right0exp]...]).
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         return self.head(x.view(x.size(0), -1))

# BATCH_SIZE = 128
# GAMMA = 0.999
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200
# TARGET_UPDATE = 10

#########
# Agent #
#########
class MyAgent(CaptureAgent):
  """
  YOUR DESCRIPTION HERE
  """
  def __init__(self, index, timeForComputing = .1):
    super(MyAgent, self).__init__(index, timeForComputing)
    self.previousScreens = [None, None]
    # self.previousScore = 0
    self.previousActions = [None, None]
    self.initialFood = None

    # self.actionSpaceSize = 5
    # self.policy_net = None
    # self.target_net = None
    # self.optimizer = None
    # self.memory = ReplayMemory()
    # self.steps_done = 0
    # self.imageHeight = 0
    # self.imageWidth = 0
    # self.doTargetNetUpdate = True if random.random() > 0.9 else False

  # def tensorToAction(self, tensor):
  #   value = tensor[0][0].item()
  #   if 0 < value <=1:
  #     return "North"
  #   elif 1 < value <=2:
  #     return "South"
  #   elif 2 < value <=3:
  #     return "East"
  #   elif 3 < value <=4:
  #     return "West"
  #   else:
      # return "Stop"

  # def updateScreenAndAction(self, screen, action):
  #   self.previousScreens[0] = self.previousScreens[1]
  #   self.previousScreens[1] = screen
  #   self.previousActions[0] = self.previousActions[1]
  #   self.previousActions[1] = action

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

    self.steps_done = 0

    # # DQN Stuff
    # self.imageHeight = gameState.getWalls().height
    # self.imageWidth = gameState.getWalls().width
    # self.policy_net = DQN(self.imageHeight, self.imageWidth, self.actionSpaceSize).to(device)
    # self.target_net = DQN(self.imageHeight, self.imageWidth, self.actionSpaceSize).to(device)
    # self.target_net.load_state_dict(self.policy_net.state_dict())
    # self.target_net.eval()
    # self.optimizer = optim.RMSprop(self.policy_net.parameters())

  # def getScreen(self, gameState): # Maybe do grayscale?
  #   image = np.zeros((self.imageWidth, self.imageHeight, 3), dtype=np.uint8)

  #   gray = (145, 145, 145)
  #   for (x, y) in gameState.getWalls().asList():
  #     image[x, y] = gray

  #   yellow = (0, 200, 255)
  #   for (x, y) in gameState.getFood().asList():
  #     image[x, y] = yellow

  #   green = (0, 255, 0)
  #   for index in gameState.getPacmanTeamIndices():
  #     position = gameState.getAgentPosition(index)
  #     if position is not None:
  #       (x, y) = position
  #       image[x, y] = green

  #   red = (0, 0, 255)
  #   for index in gameState.getGhostTeamIndices():
  #     position = gameState.getAgentPosition(index)
  #     if position is not None:
  #       (x, y) = position
  #       image[x, y] = red

  #   return image

  def chooseAction(self, gameState, policy_net, state):
    """
    Picks among actions randomly.
    """
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * self.steps_done / EPS_DECAY)
    self.steps_done += 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    legalActions = gameState.getLegalActions(self.index)
    # print("    Legal actions:")
    # for action in legalActions:
    #   print("      ", action)
    filteredActions = actionsWithoutReverse(actionsWithoutStop(legalActions), gameState, self.index)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            tensors = policy_net(state)
            # .max(1)[1].view(1, 1)
            # action1 = util.tensor_to_action(tensors[0][0])
            # tensor2 = tensors[0]

            y = tensors.view(1, -1)
            action = None
            for index in range(4, 0, -1):
              action = util.tensor_index_to_action(index)
              # print("          Checking action", action)
              if action not in filteredActions:
                # print("          Rejecting bad action {0}, index = {1}".format(action, index))
                continue
              else:
                break
            print("        PolicyNet returned action ", action)
            return action
    else:
        action = None
        while action not in filteredActions:
            tensor = torch.tensor([[random.randrange(5)]], device=device, dtype=torch.long)
            action = util.tensor_to_action(tensor[0][0])
        print("        Random returned action ", action)
        return action
    # if self.initialFood is None:
    #   self.initialFoodCount = gameState.getFood().count()
    # foodRemaining = gameState.getFood().count()

    # image = self.getScreen(gameState)
    # screen = np.ascontiguousarray(image, dtype=np.float32) / 255
    # screen = torch.from_numpy(screen)
    # screen = resize(screen).unsqueeze(0).to(device)

    # next_state = None
    # if self.steps_done >= 2:
    #   state = self.previousScreens[1] - self.previousScreens[0]
    #   next_state = screen - self.previousScreens[1]

    #   # Calculate reward
    #   # Reward = (height x width) * -remaining food - total distance to food
    #   totalfoodDistance = 0
    #   for index in gameState.getPacmanTeamIndices():
    #     foodDistance = 0
    #     if foodRemaining > 0:
    #       foodDistance = inf
    #       for foodPosition in gameState.getFood():
    #         distance = util.manhattanDistance(gameState.getAgentPosition(index), foodPosition)
    #         if distance < foodDistance:
    #           foodDistance = distance
    #     totalfoodDistance += foodDistance
    #   reward = ((self.imageHeight * self.imageWidth) * -foodRemaining) - totalfoodDistance

    #   self.memory.push(state, self.previousActions[0], next_state, reward)
    #   self.optimizeModel()

    #   if self.doTargetNetUpdate:
    #     self.target_net.load_state_dict(self.policy_net.state_dict())

    # # cv.imshow("demo", image)
    # # cv.waitKey(0)

    # teammateActions = self.receivedBroadcast
    # # Process your teammate's broadcast! 
    # # Use it to pick a better action for yourself

    # actions = gameState.getLegalActions(self.index)

    # filteredActions = actionsWithoutReverse(actionsWithoutStop(actions), gameState, self.index)

    # sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * self.steps_done / EPS_DECAY)
    # self.steps_done += 1
    # if sample > eps_threshold and self.steps_done >= 2:
    #     with torch.no_grad():
    #         # t.max(1) will return largest column value of each row.
    #         # second column on max result is index of where max element was
    #         # found, so we pick action with the larger expected reward.
    #         actions = self.policy_net(next_state)
    #         # .max(1)[1].view(1, 1)
    #         action = actions.max(1)[1].view(1, 1)
    #         self.updateScreenAndAction(screen, action)
    #         return action
    # else:
      # action = None
      # while action not in filteredActions:
      #   actionTensor = torch.tensor([[random.randrange(self.actionSpaceSize)]], device=device, dtype=torch.long)
      #   action = self.tensorToAction(actionTensor)
      # self.updateScreenAndAction(screen, actionTensor)
      # return action

  # def optimizeModel(self):
  #   if len(self.memory) < BATCH_SIZE:
  #       return
  #   transitions = self.memory.sample(BATCH_SIZE)
  #   # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
  #   # detailed explanation). This converts batch-array of Transitions
  #   # to Transition of batch-arrays.
  #   batch = Transition(*zip(*transitions))

  #   # Compute a mask of non-final states and concatenate the batch elements
  #   # (a final state would've been the one after which simulation ended)
  #   non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
  #                                         batch.next_state)), device=device, dtype=torch.uint8)
  #   non_final_next_states = torch.cat([s for s in batch.next_state
  #                                               if s is not None])
  #   state_batch = torch.cat(batch.state)
  #   action_batch = torch.cat(batch.action)
  #   reward_batch = torch.cat(batch.reward)

  #   # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
  #   # columns of actions taken. These are the actions which would've been taken
  #   # for each batch state according to policy_net
  #   state_action_values = self.policy_net(state_batch).gather(1, action_batch)

  #   # Compute V(s_{t+1}) for all next states.
  #   # Expected values of actions for non_final_next_states are computed based
  #   # on the "older" target_net; selecting their best reward with max(1)[0].
  #   # This is merged based on the mask, such that we'll have either the expected
  #   # state value or 0 in case the state was final.
  #   next_state_values = torch.zeros(BATCH_SIZE, device=device)
  #   next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
  #   # Compute the expected Q values
  #   expected_state_action_values = (next_state_values * GAMMA) + reward_batch

  #   # Compute Huber loss
  #   loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

  #   # Optimize the model
  #   self.optimizer.zero_grad()
  #   loss.backward()
  #   for param in self.policy_net.parameters():
  #       param.grad.data.clamp_(-1, 1)
  #   self.optimizer.step()

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
