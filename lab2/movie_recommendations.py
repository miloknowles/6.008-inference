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

    log_prior = np.log(prior)

    # Each column of likelihood represents a certain movie. Of the rows we care
    # about (observations), sum the log likelihoods.
    # This is the log likelihood sum for each x.
    log_likelihood_sums = np.sum(np.log(likelihood[np.array(y),:]), axis=0)

    log_marginal_y = np.log(np.sum(likelihood[np.array(y),:], axis=1))
    log_marginal_y_sum = np.sum(log_marginal_y) # Scalar value.

    # Posterior = prior * likelihood
    posterior = np.exp(log_prior + log_likelihood_sums - log_marginal_y_sum)
    posterior /= np.sum(posterior)

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

    inverse_logs = -1 * np.log2(distribution, where=(distribution != 0))
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


def main():
    # Here are some error checks that you can use to test your code.
    # print("Posterior calculation (few observations)")
    # prior = np.array([0.6, 0.4])
    # likelihood = np.array([
    #     [0.7, 0.98],
    #     [0.3, 0.02],
    # ])
    # y = [0]*2 + [1]*1
    # print("My answer:")
    # print(compute_posterior(prior, likelihood, y))
    # print("Expected answer:")
    # print(np.array([[0.91986917, 0.08013083]]))

    # print("---")
    # print("Entropy of fair coin flip")
    # distribution = np.array([0.5, 0.5])
    # print("My answer:")
    # print(compute_entropy(distribution))
    # print("Expected answer:")
    # print(1.0)

    # print("Entropy of coin flip where P(heads) = 0.25 and P(tails) = 0.75")
    # distribution = np.array([0.25, 0.75])
    # print("My answer:")
    # print(compute_entropy(distribution))
    # print("Expected answer:")
    # print(0.811278124459)

    # print("Entropy of coin flip where P(heads) = 0.75 and P(tails) = 0.25")
    # distribution = np.array([0.75, 0.25])
    # print("My answer:")
    # print(compute_entropy(distribution))
    # print("Expected answer:")
    # print(0.811278124459)

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
    # SEE test.py for additional unit tests!

    true_ratings = infer_true_movie_ratings(-1)
    print true_ratings

    entropies = compute_true_movie_rating_posterior_entropies(-1)
    print entropies

if __name__ == '__main__':
    main()
