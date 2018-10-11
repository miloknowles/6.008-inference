#!/usr/bin/env python
"""
movie_recommendations.py
Original author: Felix Sun (6.008 TA, Fall 2015)
Modified by:
- Danielle Pace (6.008 TA, Fall 2016),
- George H. Chen (6.008/6.008.1x instructor, Fall 2016)

Please read the project instructions beforehand! Your code should go in the
blocks denoted by "YOUR CODE GOES HERE" -- you should not need to modify any
other code!
"""

import matplotlib.pyplot as plt
import movie_data_helper
import numpy as np
from numpy import ma
import scipy
import scipy.misc
from sys import exit


def compute_posterior(prior, likelihood, y):
    """
    Use Bayes' rule for random variables to compute the posterior distribution
    of a hidden variable X, given N i.i.d. observations Y_0, Y_1, ..., Y_{N-1}.

    Hidden random variable X is assumed to take on a value in {0, 1, ..., M-1}.

    Each random variable Y_i takes on a value in {0, 1, ..., K-1}.

    Inputs
    ------
    - prior: a length M vector stored as a 1D NumPy array; prior[m] gives the
        (unconditional) probability that X = m
    - likelihood: a K row by M column matrix stored as a 2D NumPy array;
        likelihood[k, m] gives the probability that Y = k given X = m
    - y: a length-N vector stored as a 1D NumPy array; y[n] gives the observed
        value for random variable Y_n

    Output
    ------
    - posterior: a length M vector stored as a 1D NumPy array: posterior[m]
        gives the probability that X = m given
        Y_0 = y_0, ..., Y_{N-1} = y_{n-1}
    """
    # -------------------------------------------------------------------------
    # ERROR CHECKS -- DO NOT MODIFY
    #
    # check that prior probabilities sum to 1
    if np.abs(1 - np.sum(prior)) > 1e-06:
        exit('In compute_posterior: The prior probabilities need to sum to 1')

    # check that likelihood is specified as a 2D array
    if len(likelihood.shape) != 2:
        exit('In compute_posterior: The likelihood needs to be specified as ' +
             'a 2D array')

    K, M = likelihood.shape

    # make sure likelihood and prior agree on number of hidden states
    if len(prior) != M:
        exit('In compute_posterior: Mismatch in number of hidden states ' +
             'according to the prior and the likelihood.')

    # make sure the conditional distribution given each hidden state value sums
    # to 1
    for m in range(M):
        if np.abs(1 - np.sum(likelihood[:, m])) > 1e-06:
            exit('In compute_posterior: P(Y | X = %d) does not sum to 1' % m)

    # Use masked array to fill log0 with 0.
    log_prior = ma.log(prior).filled(0)

    # Each column of likelihood represents a certain movie. Of the rows we care
    # about (observations), sum the log likelihoods. This is the log likelihood sum for each x.
    log_likelihood_sums = np.sum(ma.log(likelihood[np.array(y),:]).filled(0), axis=0)

    log_sums = log_prior + log_likelihood_sums
    c_max = np.max(log_sums) # Find maximum log value to shift all values by (avoid underflow).

    # Shift all values by c_max will leave ratios unaffected (equivalent to multiplying) each
    # entry in posterior by exp(-c_max).
    posterior = np.exp(log_sums - c_max)
    posterior /= np.sum(posterior) # Normalize.

    return posterior


def compute_movie_rating_likelihood(M):
    """
    Compute the rating likelihood probability distribution of Y given X where
    Y is an individual rating (takes on a value in {0, 1, ..., M-1}), and X
    is the hidden true/inherent rating of a movie (also takes on a value in
    {0, 1, ..., M-1}).

    Please refer to the instructions of the project to see what the
    likelihood for ratings should be.

    Output
    ------
    - likelihood: an M row by M column matrix stored as a 2D NumPy array;
        likelihood[k, m] gives the probability that Y = k given X = m
    """
    likelihood = np.zeros((M, M))

    for y in range(M):
        for x in range(M):
            likelihood[y, x] = 1.0 / abs(y - x) if y != x else 2

    # Remember to normalize the likelihood, so that each column is a
    # probability distribution.
    for col in range(M):
        likelihood[:,col] /= np.sum(likelihood[:,col])

    return likelihood


def infer_true_movie_ratings(num_observations=-1):
    """
    For every movie, computes the posterior distribution and MAP estimate of
    the movie's true/inherent rating given the movie's observed ratings.

    Input
    -----
    - num_observations: integer that specifies how many available ratings to
        use per movie (the default value of -1 indicates that all available
        ratings will be used).

    Output
    ------
    - posteriors: a 2D array consisting of the posterior distributions where
        the number of rows is the number of movies, and the number of columns
        is M, i.e., the number of possible ratings (remember ratings are
        0, 1, ..., M-1); posteriors[i] gives a length M vector that is the
        posterior distribution of the true/inherent rating of the i-th movie
        given ratings for the i-th movie (where for each movie, the number of
        observations used is precisely what is specified by the input variable
        `num_observations`)
    - MAP_ratings: a 1D array with length given by the number of movies;
        MAP_ratings[i] gives the true/inherent rating with the highest
        posterior probability in the distribution `posteriors[i]`
    """
    M = 11  # all of our ratings are between 0 and 10
    prior = np.array([1.0 / M] * M)  # uniform distribution
    likelihood = compute_movie_rating_likelihood(M)

    # get the list of all movie IDs to process
    movie_id_list = movie_data_helper.get_movie_id_list()
    num_movies = len(movie_id_list)

    # Allocate output variables.
    posteriors = np.zeros((num_movies, M))
    MAP_ratings = np.zeros(num_movies)

    for i, movie_id in enumerate(movie_id_list):
        # Truncate the number of movies if necessary.
        ratings = movie_data_helper.get_ratings(movie_id)[:num_observations]

        # Compute the posterior probability.
        posteriors[i,:] = compute_posterior(prior, likelihood, ratings)

        # MAP Rating is simply the rating with maximum posterior probability.
        MAP_ratings[i] = np.argmax(posteriors[i,:])

    return posteriors, MAP_ratings


def compute_entropy(distribution):
    """
    Given a distribution, computes the Shannon entropy of the distribution in
    bits.

    Input
    -----
    - distribution: a 1D array of probabilities that sum to 1

    Output:
    - entropy: the Shannon entropy of the input distribution in bits
    """

    # -------------------------------------------------------------------------
    # ERROR CHECK -- DO NOT MODIFY
    #
    if np.abs(1 - np.sum(distribution)) > 1e-6:
        exit('In compute_entropy: distribution should sum to 1.')

    inverse_logs = -1 * ma.log2(distribution).filled(0)
    entropy = np.sum(distribution * inverse_logs)

    return entropy


def compute_true_movie_rating_posterior_entropies(num_observations):
    """
    For every movie, computes the Shannon entropy (in bits) of the posterior
    distribution of the true/inherent rating of the movie given observed
    ratings.

    Input
    -----
    - num_observations: integer that specifies how many available ratings to
        use per movie (the default value of -1 indicates that all available
        ratings will be used)

    Output
    ------
    - posterior_entropies: a 1D array; posterior_entropies[i] gives the Shannon
        entropy (in bits) of the posterior distribution of the true/inherent
        rating of the i-th movie given observed ratings (with number of
        observed ratings given by the input `num_observations`)
    """
    posterior_dist, _ = infer_true_movie_ratings(num_observations)

    n_dist = posterior_dist.shape[0]
    posterior_entropies = np.zeros(n_dist) # Each row is a posterior distribution.

    for i in range(n_dist):
        posterior_entropies[i] = compute_entropy(posterior_dist[i])

    return posterior_entropies


def get_top_k_movie_ids(k, best=True):
    """
    Queries movies to get the best OR worst k movies, based on args.
    """
    _, true_ratings = infer_true_movie_ratings(-1)
    num_movies = len(true_ratings)
    k = min(k, num_movies)

    # Note: this is not a stable sort, and index orders don't seem to be preserved.
    sorted_indices = np.argsort(true_ratings)

    top_k_indices = sorted_indices[-k:] if best else sorted_indices[:k]
    top_k_ratings = true_ratings[top_k_indices]

    # Get movie names.
    names = []
    for id in top_k_indices:
        names.append(movie_data_helper.get_movie_name(id))

    return top_k_indices, top_k_ratings, names


def plot_entropy_vs_num_samples():
    xs = np.arange(1, 201)
    ys = []
    for num_samples in xs:
        print('Computing for x=%d' % num_samples)
        entropies = compute_true_movie_rating_posterior_entropies(num_samples)
        avg_entropy = np.mean(entropies)
        ys.append(avg_entropy)

    plt.plot(xs, ys)
    plt.title('Average Entropy vs. Num. Observations')
    plt.xlabel('Observations')
    plt.ylabel('Average Entropy (bits)')
    plt.show()


def main():
    # Here are some error checks that you can use to test your code.
    # I moved these checks into test.py.

    #
    # END OF ERROR CHECKS
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE FOR TESTING THE FUNCTIONS YOU HAVE WRITTEN,
    # for example, to answer the questions in part (e) and part (h)
    #
    # Place your code that calls the relevant functions here.  Make sure it's
    # easy for us graders to run your code. You may want to define multiple
    # functions for each of the parts of this problem, and call them here.
    #
    # END OF YOUR CODE FOR TESTING
    # -------------------------------------------------------------------------

    ####################################
    # PART D: Infer true ratings.
    ####################################
    # posteriors, true_ratings = infer_true_movie_ratings(-1)
    # print true_ratings
    # print posteriors

    ####################################
    # PART E: Find best and worst movies.
    ####################################
    # print('Top 10 Movies (based on inferred rating)')
    # best_ids, best_ratings, best_names = get_top_k_movie_ids(40)
    # print(best_ids)
    # print(best_ratings)
    # print(best_names)

    # print('Worst 10 Movies (based on inferred rating)')
    # worst_ids, worst_ratings, worst_names = get_top_k_movie_ids(10, False)
    # print(worst_ids)
    # print(worst_ratings)
    # print(worst_names)

    ###################################
    # PART H: plot average entropy.
    ###################################
    plot_entropy_vs_num_samples()


if __name__ == '__main__':
    main()
