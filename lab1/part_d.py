# Lab 1: Part D

import numpy as np

def invest_log(f, w, N):
	"""
	f: numpy array (W x M) where W is the number of weather types, and M is the number of bonds.
	w: numpy array (N) weather sequence
	N: number of days 
	"""
	# Start with $1 in each bond. Log($1) = 0.
	log_values = np.zeros(f.shape[1])

	# Sum the logs instead of multiplying values.
	# v = (1 + r1)(1 + r2) ...
	# log(v) = log(1 + r1) + log(1 + r2) ...
	for timestep in range(N):
		weather = int(w[timestep])
		log_values += np.log(1 + f[weather,:])

	# Find the maximum log value.
	c_max = np.max(log_values)

	# Shift all values by the maximum.
	values_shifted = np.exp(log_values - c_max)

	# Exponentiate to undo logs.
	total = np.sum(values_shifted)

	return values_shifted / total
	
# Test for N = 25 with provided files.
f_in = np.loadtxt('f.txt')
w_in = np.loadtxt('w.txt')
N = 25
alloc = invest_log(f_in, w_in, N)
print('[Part D] Allocation:', alloc)
