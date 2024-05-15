# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.


def question2():
    answerDiscount = 0.9 #high discoun value
    answerNoise = 0 #reducing it to a zero value to make crossing the bridge more appealing and act as a deterrent from exploration
    return answerDiscount, answerNoise


def question3a():
    answerDiscount = 0.01 #small positive discount to choose rewards
    answerNoise = 0  #zero noise to not incentivise exploration
    answerLivingReward = 0.1  #positive living reward to encourage living
    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    answerDiscount = 0.1  #moderate discount to choose rewards
    answerNoise = 0.1  #moderate positive noise to discourage exploration
    answerLivingReward = 0.1  #positive living reward to encourage living
    return answerDiscount, answerNoise, answerLivingReward


def question3c():
    answerDiscount = 0.6  #high discount to choose rewards
    answerNoise = 0.01  #small positive noise to incentivise exploration
    answerLivingReward = 0.01  #small living reward to encourage living
    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    answerDiscount = 0.5  #high discount to choose rewards
    answerNoise = 0.1  #moderate noise to encourage safer navigation
    answerLivingReward = 0.01  #small positive living reward to encourage living
    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    answerDiscount = 0  #zero discount to prioritize long-term rewards
    answerNoise = 0  #zero noise for exploration
    answerLivingReward = 1  #positive living reward to keep the agent moving
    return answerDiscount, answerNoise, answerLivingReward

def question8():
    epsilon_values = [0] + [i / 100 for i in range(1, 101)]
    learning_rates = [0.1, 0.5, 0.9, 1]
    return 'NOT POSSIBLE'


if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
