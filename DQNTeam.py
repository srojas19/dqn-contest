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

import numpy as np

from keras.models import model_from_json, clone_model
from keras.optimizers import SGD , Adam

#################
# Team creation #
#################

ACTIONS = [Directions.STOP, Directions.NORTH, Directions.SOUTH, Directions.WEST, Directions.EAST]
LEARNING_RATE = 0.00025

IMG_ROWS = 18

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
    self.model = createCNNwithAdam(learningRate= LEARNING_RATE)

    self.model.load_weights("model.h5")
    adam = Adam(lr=LEARNING_RATE)
    self.model.compile(loss='mse',optimizer=adam)


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    '''
    You should change this in your own agent.
    '''

    image = self._createMapRepresentation(gameState, self.index)

    legalActionsVector = self._getLegalActionsVector(gameState, self.index)

    q = self.model.predict(image)
    q = q + legalActionsVector
    action_index = np.argmax(q)
    action = ACTIONS[action_index]

    return action

  def _getLegalActionsVector(self, state, agentIndex):
    legalActions = state.getLegalActions(agentIndex)
    vector = np.zeros(5)
    for i in range(vector.size):
        vector[i] = 0 if ACTIONS[i] in legalActions else -1000

    return vector

  def _createMapRepresentation(self, state, agentIndex):
    """
    Create an image representation of the state that can be sent as an input to the CNN.
    One could picture this as a simplified image of the map in a given state, but instead of using
    multiple pixels for each object in the map (that is, an agent, a wall, ...), it will be represented
    with a single, one channel (black and white), pixel. 
    """

    data = str(state.data).split("\n")
    data.pop()
    data.pop()

    representation = []
    rowIndex = 0
    for rowIndex in range(len(data)):
        representation.append([])
        for char in list(data[rowIndex]):
            representation[rowIndex].append(ord(char))

    representation = np.array(representation)

    # Colors partner
    partnerPosition = state.getAgentPosition((agentIndex + 2) % state.getNumAgents())
    representation[IMG_ROWS - partnerPosition[1] -1][partnerPosition[0]] = 180

    # Colors active agent
    agentPosition = state.getAgentPosition(agentIndex)
    representation[IMG_ROWS - agentPosition[1] -1][agentPosition[0]] = 200

    # USE THESE LINES IF YOU WANT TO CHECK THE IMAGE REPRESENTATION OF THE STATE,
    # SEEN BY THE AGENT THAT EXECUTES THE FUNCTION
    # plt.imshow(representation)
    # plt.show()

    representation = representation.reshape([1, representation.shape[0], representation.shape[1], 1])
    return representation

