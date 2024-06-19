# GridWorld with Cliff - Q-Learning vs Sarsa

This mini project demonstrates the comparison between Q-Learning and Sarsa algorithms in a GridWorld environment with a cliff.

## Description

The GridWorld environment consists of a grid where an agent starts at the bottom left corner and aims to reach the goal at the bottom right corner. However, there is a cliff between the start and the goal positions. If the agent steps into the cliff, it receives a significant negative reward and is sent back to the start position. 

### Algorithms Compared

- **Q-Learning**: An off-policy temporal difference control algorithm.
- **Sarsa**: An on-policy temporal difference control algorithm.

## Sum of Rewards during Episodes

The following plot shows the sum of rewards during episodes for both Q-Learning and Sarsa algorithms:

![Sum of Rewards during Episodes](https://github.com/tarhanefe/q-vs-sarsa-comparsion/blob/main/Q_Learning%20vs%20Sarsa.png)

## Trajectories

### Sarsa Trajectory

The trajectory followed by the agent using the Sarsa algorithm is shown below:

![Sarsa Trajectory](//Sarsa Trajectory.png)
