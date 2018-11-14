import numpy as np
import matplotlib.pyplot as plt
import capture
import game
from game import Directions

ACTIONS = [Directions.STOP, Directions.NORTH, Directions.SOUTH, Directions.WEST, Directions.EAST]

def getLegalActionsVector(state, agentIndex):
    legalActions = state.getLegalActions(agentIndex)
    vector = np.zeros(5)
    for i in range(vector.size):
        vector[i] = 0 if ACTIONS[i] in legalActions else -1000

    return vector


def createMapRepresentation(state, agentIndex):
    """
    Creates an image representation of the state that can be sent as an input to the CNN.
    One could picture this as a simplified image of the map in a given state, but instead of using
    multiple pixels for each object in the map (that is, an agent, a wall, ...), it will be represented
    with a single, one channel (black and white), pixel. 
    """
    IMG_ROWS = state.data.layout.height

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

    # COLOR AGENTS:
    for agent in range(state.getNumAgents()):
        agentPosition = state.getAgentPosition(agent)
        agentState = state.getAgentState(agent)
        if agent == agentIndex and not agentState.isPacman:
            representation[IMG_ROWS - agentPosition[1] -1][agentPosition[0]] = 200
        elif agent == agentIndex and agentState.isPacman:
            representation[IMG_ROWS - agentPosition[1] -1][agentPosition[0]] = 220    
        elif state.isOnRedTeam(agentIndex) == state.isOnRedTeam(agent):
            representation[IMG_ROWS - agentPosition[1] -1][agentPosition[0]] = 150
        else:
            representation[IMG_ROWS - agentPosition[1] -1][agentPosition[0]] = 80
        
        if agentState.scaredTimer > 0 and not agentState.isPacman:
            representation[IMG_ROWS - agentPosition[1] -1][agentPosition[0]] += 10

    # USE THESE LINES IF YOU WANT TO CHECK THE IMAGE REPRESENTATION OF THE STATE,
    # SEEN BY THE AGENT THAT EXECUTES THE FUNCTION
    # plt.imshow(representation)
    # plt.show()

    representation = representation.reshape([1, representation.shape[0], representation.shape[1], 1])
    return representation
