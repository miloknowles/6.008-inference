#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/12/2012 5:36pm
# Modified by: <your name here!>

# use this to enable/disable graphics
enable_graphics = True

import sys
import numpy as np
import robot
if enable_graphics:
    import graphics

from robot import Distribution


#-----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states

    all_possible_observed_states: a list of possible observed states

    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state

    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state

    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """
    num_time_steps = len(observations)
    n = len(all_possible_hidden_states)

    # Make a transition matrix. T[j,i] is the probability of arriving at state j
    # from state i. Therefore, each column i is a transition distribution for a
    # particular from_state i.
    T = np.zeros((n, n))

    for i, from_state in enumerate(all_possible_hidden_states):
        tmodel = transition_model(from_state) # Distribution over next states.
        T[:,i] = np.array([tmodel[next_state] for next_state in all_possible_hidden_states])

    # Make node potentials for each node x_t = 1...n
    # The node potential is the observation likelihood.
    # TODO: get rid of this double loop!!!
    node_potentials = np.zeros((num_time_steps, n))
    for t in range(0, num_time_steps):
        if (observations[t] is None):
            node_potentials[t] = np.ones(n)
        else:
            for i, node_state in enumerate(all_possible_hidden_states):
                likelihood_dist = observation_model(node_state)
                node_potentials[t,i] = likelihood_dist[observations[t]]

    # Make sure that we incorporate known action at t=0 (STAY).
    for i, state in enumerate(all_possible_hidden_states):
        if state[2] != 'stay':
            node_potentials[0, i] = 0
    node_potentials[0] /= np.sum(node_potentials[0])

    # Each forward_messages[i] stores the message from i-1 to i.
    # The message is a numpy array of length n, (where n is # hidden states).
    forward_messages = [np.zeros(n)] * num_time_steps

    # First forward message is just the prior.
    # This is uniform over all locations.
    forward_messages[0] = np.array([prior_distribution[state_i] for state_i in all_possible_hidden_states])

    # Compute forward messages.
    for t in range(1, num_time_steps):
        forward_messages[t] = np.matmul(T, forward_messages[t-1] * node_potentials[t-1])
        forward_messages[t] /= np.sum(forward_messages[t]) # Normalize.

    # Compute backward messages.
    backward_messages = [np.zeros(n)] * num_time_steps
    backward_messages[-1] = np.ones(n)
    for t in reversed(range(num_time_steps-1)):
        backward_messages[t] = np.matmul(np.transpose(T), backward_messages[t+1] * node_potentials[t+1])
        backward_messages[t] /= np.sum(backward_messages[t]) # Normalize.

    marginals = [None] * num_time_steps # remove this

    # Each marginal is the product of message before, message after, and node potential.
    for t in range(num_time_steps):
        marginals[t] = node_potentials[t].copy()
        if t > 0:
            marginals[t] *= forward_messages[t]

        marginals[t] /= np.sum(marginals[t])
        if t < (num_time_steps-1):
            marginals[t] *= backward_messages[t]

    # Convert marginals to Distribution.
    marginals_dist = []
    for marg in marginals:
        dist = Distribution()
        for i, state in enumerate(all_possible_hidden_states):
            dist[state] = marg[i]
        dist.renormalize()
        marginals_dist.append(dist)

    return marginals_dist

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: This is for you to implement

    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    return estimated_hidden_states

def second_best(all_possible_hidden_states,
                all_possible_observed_states,
                prior_distribution,
                transition_model,
                observation_model,
                observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: This is for you to implement

    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    return estimated_hidden_states


#-----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(initial_distribution, transition_model, observation_model,
                  num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from a hidden Markov model given an initial
    # distribution, transition model, observation model, and number of time
    # steps, generate samples from the corresponding hidden Markov model
    hidden_states = []
    observations  = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state       = initial_distribution().sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state   = hidden_states[-1]
        new_state    = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1: # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


#-----------------------------------------------------------------------------
# Main
#

if __name__ == '__main__':
    # flags
    make_some_observations_missing = False
    use_graphics                   = enable_graphics
    need_to_generate_data          = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(robot.initial_distribution,
                          robot.transition_model,
                          robot.observation_model,
                          num_time_steps,
                          make_some_observations_missing)

    all_possible_hidden_states   = robot.get_all_hidden_states()
    all_possible_observed_states = robot.get_all_observed_states()
    prior_distribution           = robot.initial_distribution()

    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 robot.transition_model,
                                 robot.observation_model,
                                 observations)
    print('\n')

    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print('\n')

    timestep = 0
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print('\n')

    timestep = 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print('\n')  

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               robot.transition_model,
                               robot.observation_model,
                               observations)
    print('\n')

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print('\n')

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(all_possible_hidden_states,
                                    all_possible_observed_states,
                                    prior_distribution,
                                    robot.transition_model,
                                    robot.observation_model,
                                    observations)
    print('\n')

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print('\n')

    difference = 0
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
    print("Number of differences between MAP estimate and true hidden " + \
          "states:", difference)

    difference = 0
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
    print("Number of differences between second-best MAP estimate and " + \
          "true hidden states:", difference)

    difference = 0
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
    print("Number of differences between MAP and second-best MAP " + \
          "estimates:", difference)

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()

