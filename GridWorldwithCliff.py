import numpy as np
import matplotlib.pyplot as plt
import operator


class GridWorld:

    def __init__(self):

        self.height = 4
        self.width = 12
        self.grid = np.zeros((self.height, self.width))-1

        self.current_location = (3, 0)

        self.start_location = (3, 0)
        self.end_location = (3, 11)
        self.hill_locations = [(3, i) for i in range(1, 11)]
        self.terminate_locations = [self.end_location]+self.hill_locations
        for i in range(10):
            self.grid[self.hill_locations[i][0],
                      self.hill_locations[i][1]] = -100

        self.actions = ['U', 'D', 'L', 'R']

    def get_available_actions(self):
        return self.actions

    def agent_on_map(self):
        grid = np.zeros((self.height, self.width))
        grid[self.current_location[0], self.current_location[1]] = 1
        return grid

    def get_reward(self, new_location):
        return self.grid[new_location[0], new_location[1]]

    def make_step(self, action):
        last_location = self.current_location

        if action == 'U':
            if last_location[0] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (
                    self.current_location[0] - 1, self.current_location[1])
                reward = self.get_reward(self.current_location)

        if action == 'D':
            if last_location[0] == 3:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (
                    self.current_location[0] + 1, self.current_location[1])
                reward = self.get_reward(self.current_location)

        if action == 'R':
            if last_location[1] == 11:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (
                    self.current_location[0], self.current_location[1] + 1)
                reward = self.get_reward(self.current_location)

        if action == 'L':
            if last_location[1] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = (
                    self.current_location[0], self.current_location[1] - 1)
                reward = self.get_reward(self.current_location)

        return reward

    def check_state(self):
        if self.current_location in self.terminate_locations:
            return 'TERMINAL'


class Q_Agent():
    # Intialise
    def __init__(self, environment, epsilon=0.1, alpha=0.5, gamma=0.7):
        self.environment = environment
        self.current_position = self.environment.current_location
        self.q_table = dict()  # Store all Q-values in dictionary of dictionaries
        # Loop through all possible grid spaces, create sub-dictionary for each
        for x in range(environment.height):
            for y in range(environment.width):
                # Populate sub-dictionary with zero values for possible moves
                self.q_table[(x, y)] = {'U': 0, 'D': 0, 'L': 0, 'R': 0}

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self, available_actions):
        """Returns the optimal action from Q-Value table. If multiple optimal actions, chooses random choice.
        Will make an exploratory random action dependent on epsilon."""
        if np.random.uniform(0, 1) < self.epsilon:
            action = available_actions[np.random.randint(
                0, len(available_actions))]
        else:
            q_values_of_state = self.q_table[self.environment.current_location]
            maxValue = max(q_values_of_state.values())
            action = np.random.choice(
                [k for k, v in q_values_of_state.items() if v == maxValue])

        return action

    def learn(self, old_state, reward, new_state, action):
        """Updates the Q-value table using Q-learning"""
        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[old_state][action]

        self.q_table[old_state][action] = (
            1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_q_value_in_new_state)


class SARSA_Agent():
    # Intialise
    def __init__(self, environment, epsilon=0.1, alpha=0.5, gamma=0.7):
        self.environment = environment
        self.current_position = self.environment.current_location
        self.q_table = dict()  # Store all Q-values in dictionary of dictionaries
        # Loop through all possible grid spaces, create sub-dictionary for each
        for x in range(environment.height):
            for y in range(environment.width):
                # Populate sub-dictionary with zero values for possible moves
                self.q_table[(x, y)] = {'U': 0, 'D': 0, 'L': 0, 'R': 0}

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self, available_actions):
        """Returns the optimal action from Q-Value table. If multiple optimal actions, chooses random choice.
        Will make an exploratory random action dependent on epsilon."""
        if np.random.uniform(0, 1) < self.epsilon:
            action = available_actions[np.random.randint(
                0, len(available_actions))]
        else:
            q_values_of_state = self.q_table[self.environment.current_location]
            maxValue = max(q_values_of_state.values())
            action = np.random.choice(
                [k for k, v in q_values_of_state.items() if v == maxValue])

        return action

    def learn(self, old_state, reward, new_state, new_action, old_action):
        """Updates the Q-value table using Q-learning"""
        new_q_value = self.q_table[new_state][new_action]
        current_q_value = self.q_table[old_state][old_action]

        self.q_table[old_state][old_action] = (
            1-self.alpha)*current_q_value + self.alpha * (reward + self.gamma * new_q_value)


def playSARSA(environment, agent, trials=800, max_steps_per_episode=1000, learn=False):
    """The play function runs iterations and updates Q-values if desired."""
    reward_per_episode = []  # Initialise performance log
    new_action = np.random.choice(['U', 'D', 'L', 'R'])
    for trial in range(trials):  # Run trials
        cumulative_reward = 0  # Initialise values of each game
        step = 0
        game_over = False
        while step < max_steps_per_episode and game_over != True:  # Run until max steps or until game is finished
            old_state = environment.current_location
            old_action = new_action
            reward = environment.make_step(old_action)
            new_state = environment.current_location
            new_action = agent.choose_action(environment.actions)
            if learn == True:  # Update Q-values if learning is specified
                agent.learn(old_state, reward, new_state,
                            new_action, old_action)

            cumulative_reward += reward
            step += 1

            if environment.check_state() == 'TERMINAL':  # If game is in terminal state, game over and start next trial
                environment.__init__()
                game_over = True

        # Append reward for current trial to performance log
        reward_per_episode.append(cumulative_reward)

    return reward_per_episode  # Return performance log


def playQ(environment, agent, trials=800, max_steps_per_episode=1000, learn=False):
    """The play function runs iterations and updates Q-values if desired."""
    reward_per_episode = []  # Initialise performance log
    actions_per_episode = []
    for trial in range(trials):  # Run trials
        cumulative_reward = 0  # Initialise values of each game
        step = 0
        game_over = False
        while step < max_steps_per_episode and game_over != True:  # Run until max steps or until game is finished
            old_state = environment.current_location
            action = agent.choose_action(environment.actions)
            reward = environment.make_step(action)
            new_state = environment.current_location

            if learn == True:  # Update Q-values if learning is specified
                agent.learn(old_state, reward, new_state, action)

            cumulative_reward += reward
            step += 1

            if environment.check_state() == 'TERMINAL':  # If game is in terminal state, game over and start next trial
                environment.__init__()
                game_over = True

        # Append reward for current trial to performance log
        reward_per_episode.append(cumulative_reward)

    return reward_per_episode  # Return performance log


def showRoute(states):
    board = np.zeros([4, 12])-1
    # add cliff marked as -1
    board[3, 1:11] = -100
    for i in range(0, 4):
        print('-------------------------------------------------')
        out = '| '
        for j in range(0, 12):
            token = '0'
            if board[i, j] == -100:
                token = '*'
            if (i, j) in states:
                token = 'R'
            if (i, j) == (3, 11):
                token = 'G'
            out += token + ' | '
        print(out)
    print('-------------------------------------------------')


trials = 500
epsilon = 0.1
alpha = 0.1
gamma = 1
environmentSARSA = GridWorld()
agentSARSA = SARSA_Agent(environmentSARSA, epsilon, alpha, gamma)
environmentQ = GridWorld()
agentQ = Q_Agent(environmentQ, epsilon, alpha, gamma)
reward_per_episodeSARSA = playSARSA(environmentSARSA, agentSARSA, trials, learn=True)
reward_per_episodeQ = playQ(environmentQ, agentQ, trials, learn=True)

# SARSA
ag_op = SARSA_Agent(environmentSARSA, epsilon = 0, alpha = 0.1, gamma = 1)
ag_op.q_table = agentSARSA.q_table

states = []
while 1:
    curr_state = ag_op.current_position
    print(curr_state)
    action = ag_op.choose_action(ag_op.environment.actions)
    states.append(curr_state)
    print("current position {} |action {}".format(curr_state, action))

    # next position
    ag_op.environment.make_step(action)
    ag_op.current_position = ag_op.environment.current_location

    if ag_op.current_position == (3,11):
        break
print('SARSA TRAJECTORY:')
showRoute(states)

# Q
ag_op = Q_Agent(environmentQ, epsilon = 0, alpha = 0.1, gamma = 1)
ag_op.q_table = agentQ.q_table

states = []
while 1:
    curr_state = ag_op.current_position
    print(curr_state)
    action = ag_op.choose_action(ag_op.environment.actions)
    states.append(curr_state)
    print("current position {} |action {}".format(curr_state, action))

    # next position
    ag_op.environment.make_step(action)
    ag_op.current_position = ag_op.environment.current_location

    if ag_op.current_position == (3,11):
        break
print('Q TRAJECTORY:')
showRoute(states)




plt.plot(reward_per_episodeSARSA, 'b')
plt.plot(reward_per_episodeQ, 'r')
plt.legend(["Sarsa", "Q-learning"])
plt.xlabel('Episodes')
plt.ylabel('Sum of reward during episode')
