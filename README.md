# Pacman Reinforcement Learning Project
Done as part of INFO-550- Artificial Intelligence course the the University of Arizona

![image](https://github.com/shakir-flash/Pacman-AI-Reinforcement-Learning/assets/59859522/4d259eef-3371-4a1c-bd8e-3feba43a75d5)

## Introduction

This project implements reinforcement learning algorithms to train Pac-Man agents to navigate mazes efficiently. The implemented algorithms include Value Iteration and Q-learning, which are fundamental techniques in the field of reinforcement learning.

### Value Iteration Algorithm

Value iteration is a dynamic programming algorithm used to solve Markov Decision Processes (MDPs). It iteratively computes the optimal value function \( V^* \) for all states in the environment. The algorithm involves the following steps:

1. **Initialization**: Initialize the value function \( V(s) \) arbitrarily for all states \( s \).
2. **Value Iteration**: Repeat until convergence:
   - For each state \( s \), update the value function using the Bellman equation:
     \[ V(s) \leftarrow \max_a \sum_{s'} T(s, a, s')[R(s, a, s') + \gamma V(s')] \]
   Where:
   - \( T(s, a, s') \) is the transition probability from state \( s \) to state \( s' \) under action \( a \).
   - \( R(s, a, s') \) is the immediate reward received after transitioning from state \( s \) to state \( s' \) under action \( a \).
   - \( \gamma \) is the discount factor for future rewards.
3. **Policy Extraction**: After convergence, extract the optimal policy \( \pi^* \) from the optimal value function \( V^* \) by selecting actions that maximize the expected return.

![image](https://github.com/shakir-flash/Pacman-AI-Reinforcement-Learning/assets/59859522/b893ab56-9415-4206-8677-f5833f4ae594) 

![image](https://github.com/shakir-flash/Pacman-AI-Reinforcement-Learning/assets/59859522/f50c4795-29a5-4f0b-846a-d573653b7866)



### Q-Learning Algorithm

Q-learning is a model-free reinforcement learning algorithm used to learn an optimal policy directly from experience without knowing the transition probabilities of the environment. It learns the Q-value function, which represents the expected return of taking a particular action in a given state. The algorithm involves the following steps:

1. **Initialization**: Initialize the Q-values arbitrarily for all state-action pairs.
2. **Exploration and Exploitation**: Repeat until convergence:
   - Select an action \( a \) using an exploration strategy (e.g., epsilon-greedy) based on the current Q-values.
   - Execute the action \( a \) and observe the reward \( R \) and the next state \( s' \).
   - Update the Q-value for the current state-action pair using the temporal difference (TD) error:
     \[ Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s, a)] \]
   Where:
   - \( \alpha \) is the learning rate.
   - \( \gamma \) is the discount factor for future rewards.
3. **Policy Extraction**: After convergence, extract the optimal policy \( \pi^* \) by selecting the action with the highest Q-value for each state.
![image](https://github.com/shakir-flash/Pacman-AI-Reinforcement-Learning/assets/59859522/c5053329-e48d-4617-8c1c-35897995050d)

## Files

- `valueIterationAgents.py`: Contains the implementation of a value iteration agent for solving known Markov Decision Processes (MDPs).
- `qlearningAgents.py`: Implements Q-learning agents for Gridworld, Crawler, and Pac-Man environments.
- `analysis.py`: Contains code to evaluate policies learned by the agents and provide results for analysis.

## Evaluation

The implemented agents were evaluated using the provided `analysis.py` script. This script evaluates the policies learned by the agents and provides results for analysis.

## Conclusion

This project provides hands-on experience with reinforcement learning algorithms and their application to navigation tasks in different environments. The implemented agents demonstrate the ability to learn optimal policies through exploration and exploitation of the state-action space.
