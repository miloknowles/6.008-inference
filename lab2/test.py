# Extra tests for Lab 2.

import unittest

from movie_recommendations import *

class TestMovieRatings(unittest.TestCase):
	def test_compute_movie_rating_likelihood_1x1(self):
		result = compute_movie_rating_likelihood(1)
		self.assertEqual(result[0,0], 1.0)

	def test_compute_movie_rating_likelihood_3x3(self):
		result = compute_movie_rating_likelihood(3)
		expected = np.array([[4.0/7, 1.0/4, 1.0/7], [2.0/7, 1.0/2, 2.0/7], [1.0/7, 1.0/4, 4.0/7]])
		self.assertTrue(np.allclose(result, expected))

if __name__ == '__main__':
  unittest.main()
