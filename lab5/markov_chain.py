from __future__ import division
import sys
import random
import numpy as np
from matplotlib import pyplot as plt

import util

def approx_markov_chain_steady_state(conditional_distribution, N_samples, iterations_between_samples):
    """
    Computes the steady-state distribution by simulating running the Markov
    chain. Collects samples at regular intervals and returns the empirical
    distribution of the samples.

    Inputs
    ------
    conditional_distribution : A dictionary in which each key is an state,
                               and each value is a Distribution over other
                               states.

    N_samples : the desired number of samples for the approximate empirical
                distribution

    iterations_between_samples : how many jumps to perform between each collected
                                 sample

    Returns
    -------
    An empirical Distribution over the states that should approximate the
    steady-state distribution.
    """
    empirical_distribution = util.Distribution()

    # Collect all valid states.
    states = list(conditional_distribution.keys())

    # Choose a random initial state.
    s = np.random.choice(states, p=None) # Uniform prior on states.

    for si in range(N_samples):
        if si % 100 == 0:
            print('[INFO] Generated samples %d/%d' % (si, N_samples))

        for i in range(iterations_between_samples):
            # With p=0.1, transition to a random state.
            if (random.random() <= 0.1):
                s = np.random.choice(states, p=None) # Choose uniformly.
            # With p=0.9, take a transition based on conditionals.
            else:
                s = conditional_distribution[s].sample()

        # Sample the current state.
        empirical_distribution[s] += 1.0

    # Normalize before returning.
    empirical_distribution.renormalize()

    return empirical_distribution

def compute_kl_divergence(d1, d2):
    """
    Computes the KL divergence formula.
    """
    # Must have the same alphabet to compute divergence.
    assert(d1.keys() == d2.keys())

    kld = 0
    for s in d1.keys():
        kld += d1[s] * np.log(d1[s] / d2[s])

    return kld

def plot_kl_divergence(conditional_distribution):
    """
    Compare the KL divergence between two empirical distributions generated
    using approx_markov_chain_steady_state above.

    Plot the results when using 128, 256, 512, 1024, 2048, 4096, and 8192 samples.
    """
    samples_sizes = [128, 256, 512] #, 1024, 2048, 4096, 8192]
    kld_values = []

    for N_samples in samples_sizes:
        dist1 = approx_markov_chain_steady_state(conditional_distribution, N_samples, 1000)
        dist2 = approx_markov_chain_steady_state(conditional_distribution, N_samples, 1000)
        kld = compute_kl_divergence(dist1, dist2)
        kld_values.append(kld)
        print('[INFO] N_samples=%d divergence=%f' % (N_samples, kld))

    plt.plot(samples_sizes, kld_values)
    plt.title('KL Divergence vs. # Samples')
    plt.ylabel('divergence')
    plt.xlabel('# samples')
    plt.show()

def compute_distributions(actor_to_movies, movie_to_actors):
    """
    Computes conditional distributions for transitioning
    between actors (states).

    Inputs
    ------
    actor_to_movies : a dictionary in which each key is an actor name and each
                      value is a list of movies that actor starred in

    movie_to_actors : a dictionary in which each key is a movie and each
                      value is a list of actors in that movie

    Returns
    -------
    A dictionary in which each key is an actor, and each value is a
    Distribution over other actors. The probability of transitioning
    from actor i to actor j should be proportional to the number of
    movies they starred in together.
    """
    out = {}
    for actor in actor_to_movies:
        conditional_distribution = util.Distribution()
        for movie in actor_to_movies[actor]:
            for co_star in movie_to_actors[movie]:
                conditional_distribution[co_star] += 1
        conditional_distribution.renormalize()
        out[actor] = conditional_distribution
    return out

def read_file(filename):
    """
    Reads in a file with actors and movies they starred in, and returns two
    dictionaries: one mapping actors to lists of movies they starred in, and one
    mapping movies to lists of actors that were in them.

    The file should have the following format:

    <Actor 1>
            <Movie 1 for actor 1>
            ...
    <Actor 2>
            <Movie 1 for actor 2>
            ...
    ...

    Actor lines should have no whitespace at the front, and movie lines
    must have whitespace at the front.
    """
    actor_to_movies = util.DefaultDict(lambda : [])
    movie_to_actors = util.DefaultDict(lambda : [])
    with open(filename) as f:
        for line in f:
            if line[0] != ' ':
                actor = line.strip()
            else:
                movie = line.strip()
                actor_to_movies[actor].append(movie)
                movie_to_actors[movie].append(actor)
    return (actor_to_movies, movie_to_actors)

def run_pagerank(data_filename, N_samples, iterations_between_samples):
    """
    Runs the PageRank algorithm, and returns the empirical
    distribution of the samples.

    Inputs
    ------
    data_filename : a file with actors and movies they starred in.

    N_samples : the desired number of samples for the approximate empirical
                distribution

    iterations_between_samples : how many jumps to perform between each collected
                                 sample

    Returns
    -------
    An empirical Distribution over the states that should approximate the
    steady-state distribution.
    """
    (actor_to_movies, movie_to_actors) = read_file(data_filename)
    conditional_distribution = compute_distributions(actor_to_movies,
                                                     movie_to_actors)

    steady_state = approx_markov_chain_steady_state(conditional_distribution,
                            N_samples,
                            iterations_between_samples)

    actors = actor_to_movies.keys()
    top = sorted( (((steady_state[actor]), actor) for actor in actors), reverse=True )

    values_to_show = min(20, len(steady_state))
    print("Top %d actors from empirical distribution:" % values_to_show)
    for i in range(0, values_to_show):
        print("%0.6f: %s" %top[i])

    # Compute and plot the KL divergence for varying # samples.
    print('[INFO] Plotting KL divergence.')
    plot_kl_divergence(conditional_distribution)

    return steady_state

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python markovChain.py <data file> <samples> <iterations between samples>")
        sys.exit(1)
    data_filename = sys.argv[1]
    N_samples = int(sys.argv[2])
    iterations_between_samples = int(sys.argv[3])

    run_pagerank(data_filename, N_samples, iterations_between_samples)

    
