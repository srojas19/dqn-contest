from __future__ import print_function

from models import CNN  # Create CNNs models from this import

import capture
from game import Directions

import argparse
# import skimage as skimage
# from skimage import transform, color, exposure
# from skimage.transform import rotate
# from skimage.viewer import ImageViewer
# import sys
# sys.path.append("game/")
# import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import json
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf

GAME = 'capture the flag' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 5 # number of valid actions {STOP, NORTH, SOUTH, WEST, EAST}
GAMMA = 0.95 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
STEPS = 3000000
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

ACTIONS = [Directions.STOP, Directions.NORTH, Directions.SOUTH, Directions.WEST, Directions.EAST]

def trainNetwork(model,args):
    
    options = capture.readCommand(['-l', 'RANDOM', '-Q'])
    game = newGame(**options)

    x_t = game.state
    s_t = createMapRepresentation(x_t, 0)

    # store the previous observations in replay memory
    D = deque()
    D2 = deque() # Separate replay memories for each agent?

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

    agentIndex = 0
    t = 0
    while t < EXPLORE:
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = Directions.STOP
        
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t = ACTIONS[action_index]
            else:
                q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                action_index = np.argmax(q)
                a_t = ACTIONS[action_index]

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

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")

def playGame(args):
    model = CNN(LEARNING_RATE)
    trainNetwork(model,args)

def newGame(layouts, agents, display, length, numGames, record, numTraining, redTeamName, blueTeamName, muteAgents=False, catchExceptions=False):
    rules = capture.CaptureRules()
    layout = layouts[0]
    
    import textDisplay
    gameDisplay = textDisplay.NullGraphics()
    rules.quiet = True
      
    game = rules.newGame( layout, agents, gameDisplay, length, muteAgents, catchExceptions )
    return game

def getSuccesor(game, state, agentIndex, action):
    newState = state.generateSuccessor(agentIndex, action)
    game.display.update(newState.data)

    reward = newState.data.scoreChange
    terminal = newState.data.timeLeft <= 0

    return newState, reward, terminal

def createMapRepresentation(state, agentIndex):
    """
    This is meant to create a representation of the state that can be sent as an input to the CNN.
    One could picture this as a simplified image of the map in a given state, but instead of using
    multiple pixels for each object in the map (that is, an agent, a wall, ...), it will be represented
    with a number. 

    """
    return []

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()