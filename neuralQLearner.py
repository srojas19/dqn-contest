class NQL:

    def __init__(self, trainAgent, model=None, modelFile=None):

        self.trainAgent = trainAgent
        self.model = model

    def reset(self, state):
        pass

    def preprocess(self, rawState):
        pass

    def getQUpdate(self, args):
        pass

    def qLearnMinibatch(self):
        pass
    
    def perceive(self, reward, rawState, terminal, testing, testingEpsilon):
        pass

    def eGreedy(self, state, testingEpsilon):
        pass

    def createNetwork(self, model):
        pass

    def loadNetwork(self, file):
        pass

    def saveNetwork(self, file):
        pass
    
