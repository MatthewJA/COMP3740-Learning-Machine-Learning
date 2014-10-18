import numpy, random

def get_states(agent, environment, epsilon=0.1):
    """
    Move through the environment according to the agent, and return tuples of the form
    [state, action, discounted_future_reward].

    agent: An object with a get_expected_rewards method as described in
        get_action, and a gamma property describing how fast discounted future
        reward falls off.
    environment: A Maze object.
    epsilon: Percentage chance of taking a random action instead of performing
        the action given by the agent. Optional (default 0.1).
    """

    environment.reset()

    lists = []
    while not environment.finished():
        state = environment.get_state()
        action = get_action(agent, environment)
        if random.random() < epsilon:
            action = random.randrange(4)
        environment.move(action)
        reward = environment.get_reward() # Temporarily - we'll update this later to be discounted
                   # future reward.
        lists.append([state, action, reward])
    if lists:
        lists[-1][2] = 1 # Solving the environment gives us reward of 1.

    tuples = []
    last_reward = 0
    while lists:
        state, action, reward = lists.pop()
        discounted_future_reward = (last_reward * agent.gamma + reward)
        last_reward = discounted_future_reward
        tuples.append((state, action, discounted_future_reward))

    return tuples

def get_action(agent, environment):
    """
    Get an action to take based on the state of the environment.

    agent: An object with a get_expected_rewards method, that takes a state
        vector (2d NumPy array with one row) and returns a NumPy array of
        expected rewards.
    """

    state = numpy.asarray([environment.get_state()])
    expected_rewards = agent.get_expected_rewards(state)
    action = numpy.argmax(expected_rewards)
    return action