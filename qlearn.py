from __future__ import print_function

from models import createCNNwithRMSProp, createCNNwithAdam  # Create CNNs models from this import

import distutils.dir_util

import sys
import time
import capture
from game import Directions

import argparse

import matplotlib.pyplot as plt
import random
import numpy as np
from collections import deque

import json
from keras.models import model_from_json, clone_model

from keras.optimizers import SGD , Adam
# import tensorflow as tf

GAME = 'capture the flag' # the name of the game being played for log files
CONFIG = 'nothreshold'

ACTIONS = [Directions.STOP, Directions.NORTH, Directions.SOUTH, Directions.WEST, Directions.EAST]
BATCH = 32 # size of minibatch
REPLAY_MEMORY = 500000 # number of previous transitions to remember
TARGET_NETWORK_UPDATE_FREQUENCY = 10000
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 50000 # timesteps to observe before training
EXPLORE = 1000000 # frames over which to anneal epsilon
TRAIN = 1500000 # Limit of training steps. TRAIN - EXPLORE = # of steps with epsilon = FINAL_EPSILON
INITIAL_EPSILON = 1 # starting value of epsilon WAS 0.1
FINAL_EPSILON = 0.1 # final value of epsilon
FRAME_PER_ACTION = 1
# LEARNING_RATE = 1e-4
LEARNING_RATE = 0.00025

def trainNetwork(model, args, options):

    targetModel = clone_model(model)
    targetModel.set_weights(model.get_weights())
    
    game = newGame(**options)

    path = "models/" + args["name"] + "/"
    distutils.dir_util.mkpath(path)

    statsFile = open(path + "stats.csv", "w")
    gamesFile = open(path + "games.csv", "w")

    # store the previous observations in replay memory
    D = deque()

    if args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    agentIndex = game.startingIndex

    s_t = game.state
    phi_t = createMapRepresentation(s_t, agentIndex)

    t = 0

    totalReward = 0
    totalGamesScore = 0
    currentGame = 1

    while t < TRAIN:
            
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = Directions.STOP

        # Choose an action epsilon greedy
        if random.random() <= epsilon or t <= OBSERVE:
            legalActions = s_t.getLegalActions(agentIndex)
            index = random.randrange(len(legalActions))
            action_index = ACTIONS.index(legalActions[index])
            a_t = ACTIONS[action_index]
            # a_t = game.agents[agentIndex].getAction(s_t)
            # action_index = ACTIONS.index(a_t)

        else:
            legalActionsVector = getLegalActionsVector(s_t, agentIndex)
            q = model.predict(phi_t)
            q = q + legalActionsVector
            action_index = np.argmax(q)
            a_t = ACTIONS[action_index]

        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        s_t1, r_t, terminal_t1 = getSuccesor(game, s_t, agentIndex, a_t)
        phi_t1 = createMapRepresentation(s_t1, agentIndex)

        # store the transition in D
        D.append((phi_t, action_index, r_t, phi_t1, terminal_t1))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            #Now we do the experience replay
            phi_j, a_j, r_j, phi_j1, terminal_j1 = zip(*minibatch)
            phi_j = np.concatenate(phi_j)
            phi_j1 = np.concatenate(phi_j1)
            y_j = targetModel.predict(phi_j) # Check this, as it might be refeeding information
            Q_sa = targetModel.predict(phi_j1)
            y_j[range(BATCH), a_j] = r_j + GAMMA*np.max(Q_sa, axis=1)*np.invert(terminal_j1)

            loss += model.train_on_batch(phi_j, y_j)

        s_t = s_t1
        phi_t = phi_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            print("Now we save model")
            model.save_weights(path + "model.h5", overwrite=True)
            with open(path + "model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        totalReward += r_t
        statsFile.write("".join([str(s) + ", " for s in [t, state, epsilon, action_index, r_t, totalReward, np.max(Q_sa), loss]]) + "\n")

        # Start new game if the last one ends.
        if game.gameOver:
            finalScore = s_t.getScore() if s_t.isOnRedTeam(agentIndex) else - s_t.getScore()
            totalGamesScore += finalScore
            gamesFile.write("".join([str(s) + ", " for s in [state, currentGame, finalScore, totalGamesScore, s_t.isOnRedTeam(agentIndex)]]) + "\n")

            currentGame += 1
            game.display.finish()
            game = newGame(**options)
            s_t = game.state
            agentIndex = game.startingIndex
            phi_t = createMapRepresentation(s_t, agentIndex)

        if t % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            targetModel = clone_model(model)
            targetModel.set_weights(model.get_weights())

    print("Episode finished!")
    print("************************")

    gamesFile.close()
    statsFile.close()

def playGame(args):
    if args["layout"] == "Default":
        options = capture.readCommand(['-Q'])
    else:
        options = capture.readCommand(['-l', 'RANDOM', '-Q'])
    startTime = time.clock()
    game = newGame(**options)
    dimensions = (game.state.data.layout.height, game.state.data.layout.width, 1)
    model = createCNNwithAdam(LEARNING_RATE, inputDimensions=dimensions)
    trainNetwork(model,args, options)

    finalTime = time.clock()
    print(finalTime - startTime)

def newGame(layouts, agents, display, length, numGames, record, numTraining, redTeamName, blueTeamName, muteAgents=False, catchExceptions=False):
    rules = capture.CaptureRules()
    layout = layouts[0]
    
    import textDisplay
    gameDisplay = textDisplay.NullGraphics()
    rules.quiet = True
      
    game = rules.newGame( layout, agents, gameDisplay, length, muteAgents, catchExceptions )

    game.display.initialize(game.state.data)
    game.numMoves = 0

    # Instructions required to register Initial state of the agents. See Game.run() for more details. 
    for i in range(len(game.agents)):
        agent = game.agents[i]
        if not agent:
            game.mute(i)
            # this is a null agent, meaning it failed to load
            # the other team wins
            print >>sys.stderr, "Agent %d failed to load" % i
            game.unmute()
            game._agentCrash(i, quiet=True)
            return
        if ("registerInitialState" in dir(agent)):
            game.mute(i)
            agent.registerInitialState(game.state.deepCopy())
            game.unmute()

    return game

def getLegalActionsVector(state, agentIndex):
    legalActions = state.getLegalActions(agentIndex)
    vector = np.zeros(5)
    for i in range(vector.size):
        vector[i] = 0 if ACTIONS[i] in legalActions else -1000

    return vector

def getSuccesor(game, state, agentIndex, action):
    """
    Return the succesor of a state, given that the agent with index 'agentIndex'
    moves in the direction stated by 'action', and all the following agents do 
    their chosen action.
    """

    game.moveHistory.append((agentIndex, action))

    moveMotivation = 0
    if action == Directions.STOP:
        moveMotivation -= 1

    newState = state.generateSuccessor(agentIndex, action)
    game.state = newState
    game.display.update(game.state.data)
    game.rules.process(game.state, game)
    
    reward = newState.data.scoreChange
    terminal = game.gameOver

    currentAgentIndex = (agentIndex + 1) % newState.getNumAgents()
    while not terminal and currentAgentIndex != agentIndex:
        action = game.agents[currentAgentIndex].getAction(newState)
        game.moveHistory.append((currentAgentIndex, action))
        newState = newState.generateSuccessor(currentAgentIndex, action)
        game.state = newState
        game.display.update(game.state.data)
        game.rules.process(game.state, game)
        reward += newState.data.scoreChange
        terminal = game.gameOver
        currentAgentIndex = (currentAgentIndex + 1) % newState.getNumAgents()

    if not newState.isOnRedTeam(agentIndex):
        reward = -reward


    # Consider foor captured
    redFoodDelta = len(list(filter(lambda x: x is True, newState.getRedFood()))) - len(list(filter(lambda x: x is True, state.getRedFood())))
    blueFoodDelta = len(list(filter(lambda x: x is True, newState.getBlueFood()))) - len(list(filter(lambda x: x is True, state.getBlueFood())))
    if newState.isOnRedTeam(agentIndex):
        reward += redFoodDelta
        reward -= blueFoodDelta
    else:
        reward -= redFoodDelta
        reward += blueFoodDelta
    
    # Promote the trained agents to move
    reward += moveMotivation

    return newState, reward, terminal

def createMapRepresentation(state, agentIndex):
    """
    Create an image representation of the state that can be sent as an input to the CNN.
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

    # # OLD METHOD TO COLOR AGENTS:
    # # Colors partner
    # partnerPosition = state.getAgentPosition((agentIndex + 2) % state.getNumAgents())
    # representation[IMG_ROWS - partnerPosition[1] -1][partnerPosition[0]] = 180

    # # Colors active agent
    # agentPosition = state.getAgentPosition(agentIndex)
    # representation[IMG_ROWS - agentPosition[1] -1][agentPosition[0]] = 200

    # NEW METHOD TO COLOR AGENT:
    for agent in range(state.getNumAgents()):
        agentPosition = state.getAgentPosition(agent)
        if agent == agentIndex:
            representation[IMG_ROWS - agentPosition[1] -1][agentPosition[0]] = 200
        elif state.isOnRedTeam(agentIndex) == state.isOnRedTeam(agent):
            representation[IMG_ROWS - agentPosition[1] -1][agentPosition[0]] = 180
        else:
            representation[IMG_ROWS - agentPosition[1] -1][agentPosition[0]] = 80
        
        if state.getAgentState(agent).scaredTimer > 0:
            representation[IMG_ROWS - agentPosition[1] -1][agentPosition[0]] += 10

    # USE THESE LINES IF YOU WANT TO CHECK THE IMAGE REPRESENTATION OF THE STATE,
    # SEEN BY THE AGENT THAT EXECUTES THE FUNCTION
    # plt.imshow(representation)
    # plt.show()

    representation = representation.reshape([1, representation.shape[0], representation.shape[1], 1])
    return representation

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    parser.add_argument('-n','--name', help='Name of the training model to train', required=True)
    parser.add_argument('-l','--layout', help='Default / Random', required=False)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    main()