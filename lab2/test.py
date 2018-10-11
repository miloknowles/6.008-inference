# Extra tests for Lab 2.

import unittest

from movie_recommendations import *

class TestInferenceMethods(unittest.TestCase):
  def test_compute_movie_rating_likelihood_1x1(self):
    result = compute_movie_rating_likelihood(1)
    self.assertEqual(result[0,0], 1.0)

  def test_compute_movie_rating_likelihood_3x3(self):
    result = compute_movie_rating_likelihood(3)
    expected = np.array([[4.0/7, 1.0/4, 1.0/7], [2.0/7, 1.0/2, 2.0/7], [1.0/7, 1.0/4, 4.0/7]])
    self.assertTrue(np.allclose(result, expected))

  def test_compute_posterior_01(self):
    prior = np.array([0.6, 0.4])
    likelihood = np.array([
        [0.7, 0.98],
        [0.3, 0.02],
    ])
    y = [0]*2 + [1]*1
    result = compute_posterior(prior, likelihood, y)
    expected = np.array([[0.91986917, 0.08013083]])
    self.assertTrue(np.allclose(result, expected, 1e-5))

  def test_infer_true_movie_ratings(self):
    pass

  def test_compute_true_movie_rating_posterior_entropies(self):
    pass

class TestEntropy(unittest.TestCase):
  def test_coin_flip(self):
    distribution = np.array([0.5, 0.5])
    result = compute_entropy(distribution)
    self.assertAlmostEqual(result, 1.0)

  def test_deterministic_coin_flip(self):
    distribution = np.array([1.0, 0.0])
    result = compute_entropy(distribution)
    self.assertAlmostEqual(result, 0.0)

  def test_4_sided_die(self):
    distribution = np.array([0.25, 0.25, 0.25, 0.25])
    result = compute_entropy(distribution)
    self.assertAlmostEqual(result, 2.0)

  def test_biased_coin_01(self):
    distribution = np.array([0.25, 0.75])
    result = compute_entropy(distribution)
    expected = 0.811278124459
    self.assertAlmostEqual(result, expected)

  def test_biased_coin_02(self):
    distribution = np.array([0.75, 0.25])
    result = compute_entropy(distribution)
    expected = 0.811278124459
    self.assertAlmostEqual(result, expected)

class TestMovieRatingInference(unittest.TestCase):
  def test_reasonable_posteriors_and_ratings(self):
    """
    Get true ratings for the entire dataset and make sure they are in range.
    """
    posteriors, true_ratings = infer_true_movie_ratings(5)

    # Make sure all posteriors sum to one.
    for i in range(posteriors.shape[0]):
      self.assertAlmostEqual(np.sum(posteriors[i,:]), 1.0)

    # Make sure all ratings are in range.
    for rating in true_ratings:
      self.assertTrue(rating <= 10 and rating >= 0)

  def test_reasonable_entropies(self):
    entropies = compute_true_movie_rating_posterior_entropies(10)
    for e in entropies:
      self.assertTrue(e >= 0 and e <= np.log2(10))

if __name__ == '__main__':
  unittest.main()
