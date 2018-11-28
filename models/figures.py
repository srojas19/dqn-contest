import sys
import distutils.dir_util
import matplotlib.pyplot as plt
import numpy as np

GAME_SCORES = 1
REWARDS = 2
Q_VALUES = 3
GAME_RESULTS = 4
LOSS = 5
AVERAGE_SCORE = 6

def main():

    path = "figures/all2/" 
    distutils.dir_util.mkpath(path)

    models = sys.argv[1:]

    figures = ['game_scores', 'rewards', 'q_values', 'game_results', 'loss', 'average_scores']

    # Set up Figures
    plt.figure(GAME_SCORES)
    plt.title('Sum of Game Scores')
    plt.xlabel('Game')
    plt.ylabel('Game Scores')
    plt.grid(True)

    plt.figure(REWARDS)
    plt.title('Sum of Rewards')
    plt.xlabel('Step')
    plt.ylabel('Rewards')
    plt.grid(True)

    plt.figure(Q_VALUES)
    plt.title('Average Q Value')
    plt.xlabel('Step')
    plt.ylabel('Q Value')
    plt.grid(True)
    
    plt.figure(GAME_RESULTS)
    plt.title('Results for the training games')
    plt.xlabel('Game')
    plt.ylabel('Total')
    plt.grid(True)

    plt.figure(LOSS)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.figure(AVERAGE_SCORE)
    plt.title('Average Score')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.grid(True)

    statsNames = ['t', 'state', 'epsilon', 'action_index', 'r_t', 'totalReward', 'Q_sa', 'loss']
    gamesNames = ['state', 'currentGame', 'finalScore', 'totalGamesScore', 'playedRed']
    for model in models:
        statsPath = model + '/stats.csv'
        gamesPath = model + '/games.csv'

        stats = np.genfromtxt(statsPath, dtype=None, delimiter=',', usecols=range(8), names=statsNames)
        games = np.genfromtxt(gamesPath, dtype=None, delimiter=',', usecols=range(4), names=gamesNames)

        group = 500
        t = np.arange(0,1500000, group)


        plt.figure(GAME_SCORES)
        plt.plot(games['currentGame'], games['totalGamesScore'], label=model)

        plt.figure(REWARDS)
        rewards = stats['totalReward'].reshape(-1, group).mean(axis=1)
        plt.plot(t, rewards, label=model)

        plt.figure(Q_VALUES)
        q_sa = stats['Q_sa'].reshape(-1, group).mean(axis=1)
        plt.plot(t, q_sa, label=model)

        plt.figure(GAME_RESULTS)
        loses, draws, wins = countResults(games)
        plt.plot(games['currentGame'], loses, label='Loses for ' + model)
        plt.plot(games['currentGame'], draws, label='Draws for ' + model)
        plt.plot(games['currentGame'], wins, label='Wins for ' + model)

        plt.figure(LOSS)
        loss = stats['loss'].reshape(-1, group).mean(axis=1)
        plt.plot(t, loss, label=model)

        plt.figure(AVERAGE_SCORE)
        averageScoreGroup = 50
        gamesGroup = np.arange(0,games['currentGame'].size-averageScoreGroup, averageScoreGroup)
        averageScores = games['finalScore'][:averageScoreGroup*gamesGroup.size].reshape(-1, averageScoreGroup).mean(axis=1)
        plt.plot(gamesGroup, averageScores, label=model)


    for i in range(1,7):
        plt.figure(i)
        plt.legend()
        plt.savefig(path+figures[i-1]+'.png', bbox_inches='tight')
    
    # plt.show()


'''
def main():
    plt.figure(1)                # the first figure
    plt.subplot(211)             # the first subplot in the first figure
    plt.plot([1, 2, 3])
    plt.subplot(212)             # the second subplot in the first figure
    plt.plot([4, 5, 6])


    plt.figure(2)                # a second figure
    plt.plot([4, 5, 6])          # creates a subplot(111) by default

    plt.figure(1)                # figure 1 current; subplot(212) still current
    plt.subplot(211)             # make subplot(211) in figure1 current
    plt.title('Easy as 1, 2, 3') # subplot 211 title

    plt.show()
'''

def countResults(games):
    scores = games['finalScore']
    numGames = scores.size

    loses = np.zeros(numGames)
    draws = np.zeros(numGames)
    wins = np.zeros(numGames)

    score = scores[0]
    for i in range(numGames):
        score = scores[i]
        if score > 0:
            loses[i] = loses[i-1]
            draws[i] = draws[i-1]
            wins[i] = wins[i-1] + 1
        elif score < 0:
            loses[i] = loses[i-1] + 1
            draws[i] = draws[i-1]
            wins[i] = wins[i-1]
        else:
            loses[i] = loses[i-1]
            draws[i] = draws[i-1] + 1
            wins[i] = wins[i-1]
    return loses, draws, wins


if __name__ == "__main__":
    main()



