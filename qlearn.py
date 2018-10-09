from __future__ import print_function

from models import CNN
from models import createCNN  # Create CNNs models from this import

import sys
import capture
from game import Directions

import argparse

import matplotlib.pyplot as plt
import random
import numpy as np
from collections import deque

import json
from keras.models import model_from_json

from keras.optimizers import SGD , Adam
# import tensorflow as tf

GAME = 'capture the flag' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 5 # number of valid actions {STOP, NORTH, SOUTH, WEST, EAST}
GAMMA = 0.95 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
STEPS = 3000000
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.9 # starting value of epsilon WAS 0.1
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

IMG_ROWS = 18
IMG_COLS = 34

ACTIONS = [Directions.STOP, Directions.NORTH, Directions.SOUTH, Directions.WEST, Directions.EAST]

def trainNetwork(model,args):
    
    # options = capture.readCommand(['-Q'])
    options = capture.readCommand(['-l', 'RANDOM', '-Q'])
    game = newGame(**options)

    x_t = game.state
    s_t = createMapRepresentation(x_t, 0)

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
    t = 0
    while t < EXPLORE:

        if x_t.isOnRedTeam(agentIndex):
            
            loss = 0
            Q_sa = 0
            action_index = 0
            r_t = 0
            a_t = Directions.STOP

            legalActionsVector = getLegalActionsVector(x_t, agentIndex)
            
            #choose an action epsilon greedy
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    # print("----------Random Action----------")
                    # legalActions = x_t.getLegalActions(agentIndex)
                    # index = random.randrange(len(legalActions))

                    # action_index = ACTIONS.index(legalActions[index])
                    # a_t = ACTIONS[action_index]
                    a_t = game.agents[agentIndex].getAction(x_t)
                    action_index = ACTIONS.index(a_t)

                    # print("action was epsilon", str(a_t), str(ACTIONS[action_index]))

                else:
                    q = model.predict(s_t)
                    q = q + legalActionsVector
                    action_index = np.argmax(q)
                    a_t = ACTIONS[action_index]
                    # print("action was predicted from the model")

            #We reduced the epsilon gradually
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            #run the selected action and observed next state and reward
            x_t1, r_t, terminal = getSuccesor(game, x_t, agentIndex, a_t)
            s_t1 = createMapRepresentation(x_t1, agentIndex)

            # store the transition in D
            D.append((s_t, action_index, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            #only train if done observing
            if t > OBSERVE:
                #sample a minibatch to train on
                minibatch = random.sample(D, BATCH)

                #Now we do the experience replay
                state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
                state_t = np.concatenate(state_t)
                state_t1 = np.concatenate(state_t1)
                targets = model.predict(state_t)
                Q_sa = model.predict(state_t1)
                targets[range(BATCH), action_t] = reward_t + GAMMA*np.max(Q_sa, axis=1)*np.invert(terminal)

                loss += model.train_on_batch(state_t, targets)

            x_t = x_t1
            s_t = s_t1
            t += 1


            # save progress every 10000 iterations
            if t % 1000 == 0:
                print("Now we save model")
                model.save_weights("model.h5", overwrite=True)
                with open("model.json", "w") as outfile:
                    json.dump(model.to_json(), outfile)

            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"

            if r_t != 0:
                print("TIMESTEP", t, "/ STATE", state, \
                    "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
                    "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

        else:
            action = game.agents[agentIndex].getAction(x_t)
            x_t, _, terminal = getSuccesor(game, x_t, agentIndex, action)

        agentIndex = (agentIndex + 1) % x_t.getNumAgents()

        # TODO: Start new game if the last one ends.
        if game.gameOver:
            game.display.finish()
            game = newGame(**options)
            x_t = game.state
            s_t = createMapRepresentation(x_t, 0)
            agentIndex = game.startingIndex


    print("Episode finished!")
    print("************************")

def playGame(args):
    model = createCNN(LEARNING_RATE)
    trainNetwork(model,args)

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
    moves in the direction stated by 'action'
    """

    # TODO: Might need to return a new game object 
    # (depends on if python modifies de argument of the function or not)

    game.moveHistory.append((agentIndex, action))
    newState = state.generateSuccessor(agentIndex, action)
    game.state = newState
    game.display.update(game.state.data)

    game.rules.process(game.state, game)

    reward = newState.data.scoreChange
    # terminal = newState.data.timeLeft <= 0
    terminal = game.gameOver

    if reward != 0:
        print("Reward displayed in getSuccesor:", reward)

    return newState, reward, terminal

def createMapRepresentation(state, agentIndex):
    """
    This is meant to create a representation of the state that can be sent as an input to the CNN.
    One could picture this as a simplified image of the map in a given state, but instead of using
    multiple pixels for each object in the map (that is, an agent, a wall, ...), it will be represented
    with a number. 

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

    #TODO: Differenciate agents
    representation = np.array(representation)

    # Colors active agent
    agentPosition = state.getAgentPosition(agentIndex)
    representation[IMG_ROWS - agentPosition[1] -1][agentPosition[0]] = 200

    # Colors partner
    partnerPosition = state.getAgentPosition(0) if agentIndex != 0 else state.getAgentPosition(2)
    representation[IMG_ROWS - partnerPosition[1] -1][partnerPosition[0]] = 180


    # USE THESE LINES IF YOU WANT TO CHECK THE IMAGE REPRESENTATION OF THE STATE,
    # SEEN BY THE AGENT THAT EXECUTES THE FUNCTION
    # plt.imshow(representation)
    # plt.show()

    representation = representation.reshape([1, representation.shape[0], representation.shape[1], 1])
    return representation

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # # sess = tf.Session(config=config)
    # from keras import backend as K
    # K.set_session(sess)
    main()