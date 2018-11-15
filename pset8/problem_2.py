import numpy as np
from matplotlib import pyplot as plt

def part_a():
	n = 100
	p = 0.2
	trials = 10000
	epsilon = 0.05

	num_bad_trials = 0
	for t in range(trials):
		S_n = np.random.binomial(n, p)

		if abs((S_n / n) - p) > epsilon:
			num_bad_trials += 1

	frac_bad_trials = num_bad_trials / trials
	print('Fraction of trials with error > epsilon:', frac_bad_trials)

def part_d():
	p = 0.2
	trials = 10000
	epsilon = 0.05

	ns = np.arange(100, 1001)
	frac = []

	for n in ns:
		print('Computing for n=%d' % n)
		num_bad_trials = 0
		for t in range(trials):
			S_n = np.random.binomial(n, p)
			if abs((S_n / n) - p) > epsilon:
				num_bad_trials += 1
		frac.append(num_bad_trials / trials)

	fig = plt.figure()
	plt.plot(ns, frac, 'ro')
	plt.xlabel('n')
	plt.ylabel('%')
	plt.title('Percent of Trials with Error > Epsilon')
	plt.show()

part_d()
