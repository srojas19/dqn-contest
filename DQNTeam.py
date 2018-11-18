# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

from models import createCNNwithAdam
from qlearnFunctions import ACTIONS, createMapRepresentation, getLegalActionsVector

import numpy as np

from keras.models import model_from_json, clone_model
from keras.optimizers import SGD , Adam

#################
# Team creation #
#################

LEARNING_RATE = 0.00025

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DQNAgent', second = 'DQNAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DQNAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
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

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    
    # modelPath = "baseline_actions_no_stopping"
    # modelPath = "random_actions_default_layout"
    # modelPath = "baseline_actions_default_layout"
    # modelPath = "random_actions_default_layout_reward_for_food"
    # modelPath = "baseline_actions_default_layout_reward_for_food"
    # modelPath = "baseline_actions_default_layout_reward_for_food_FIXED"
    # modelPath = "random_actions_default_layout_reward_for_food_FIXED"
    modelPath = "random_actions_differenciates_pacman"
    # modelPath = "baseline_actions_differenciates_pacman2"

    dimensions = (gameState.data.layout.height, gameState.data.layout.width, 1)

    self.model = createCNNwithAdam(learningRate= LEARNING_RATE, inputDimensions=dimensions)

    self.model.load_weights("models/"+ modelPath + "/model.h5")
    adam = Adam(lr=LEARNING_RATE)
    self.model.compile(loss='mse',optimizer=adam)



  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    '''
    You should change this in your own agent.
    '''

    image = createMapRepresentation(gameState, self.index)

    legalActionsVector = getLegalActionsVector(gameState, self.index)

    q = self.model.predict(image)
    q = q + legalActionsVector
    action_index = np.argmax(q)
    action = ACTIONS[action_index]

    return action