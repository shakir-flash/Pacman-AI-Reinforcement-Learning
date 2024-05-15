# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for i in range(self.iterations):
            new_values = util.Counter() #counter to store updated value for state
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state): #terminal check
                    new_values[state] = 0  #terminal states have a value of zero
                else:
                    #finding max Q-value
                    max_q_value = max([self.computeQValueFromValues(state, action) 
                                    for action in self.mdp.getPossibleActions(state)])
                    new_values[state] = max_q_value #updating current state with max q value
            self.values = new_values #updating agent value with new score

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        q_value = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            #calculating immediate reward
            reward = self.mdp.getReward(state, action, next_state)           
            #calculating the discounted future value
            discount_next_state = self.discount * self.getValue(next_state)          
            #updating Q-value
            q_value += prob * (reward + discount_next_state)        
        return q_value
    
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        max_q_value = float('-inf')
        best_action = None
        for action in self.mdp.getPossibleActions(state):
            #computing the Q-value for action in current state
            q_value = self.computeQValueFromValues(state, action)            
            #updating the best action and q-value
            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action        
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()  #get all states
        num_states = len(states)  #number of states        
        for i in range(self.iterations):
            state_index = i % num_states  #cycle through states list
            state = states[state_index]  #get current state           
            #terminal check and find if it has available actions
            if not self.mdp.isTerminal(state) and self.mdp.getPossibleActions(state):
                #update the value of the current state using the max Q-value
                self.values[state] = max(
                    self.computeQValueFromValues(state, action)
                    for action in self.mdp.getPossibleActions(state)
                )


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #computing predecessors of all states
        predecessors_map = {}
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                predecessors_map[state] = set()

        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        if next_state in predecessors_map:
                            predecessors_map[next_state].add(state)

        #initializing an empty priority queue
        priority_queue = util.PriorityQueue()

        #for each non-terminal state, computing the max q-value and push into queue
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                max_q_value = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
                diff = abs(self.values[state] - max_q_value)
                priority_queue.push(state, -diff)

        #updating states values from the priority queue
        for i in range(self.iterations):
            if priority_queue.isEmpty():
                break
            current_state = priority_queue.pop()
            #updating the value of the current state if not terminal
            if not self.mdp.isTerminal(current_state):
                self.values[current_state] = max([self.getQValue(current_state, action) for action in self.mdp.getPossibleActions(current_state)])

            #for each predecessor of the current state, update the queue
            for predecessor in predecessors_map[current_state]:
                if not self.mdp.isTerminal(predecessor):
                    max_q_value = max([self.getQValue(predecessor, action) for action in self.mdp.getPossibleActions(predecessor)])
                    diff = abs(self.values[predecessor] - max_q_value)
                    if diff > self.theta:
                        priority_queue.update(predecessor, -diff)