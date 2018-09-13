# Lab 1: Part B

import numpy as np

def invest(f, w, N):
	"""
	f: numpy array (W x M) where W is the number of weather types, and M is the number of bonds.
	w: numpy array (N) weather sequence
	N: number of days 
	"""
	# Start with $1 in each bond.
	values = np.ones(f.shape[1])

	for timestep in range(N):
		weather = int(w[timestep])
		values *= (1 + f[weather,:])

	return values / np.sum(values)
	
# Test for N = 25 with provided files.
f_in = np.loadtxt('f.txt')
w_in = np.loadtxt('w.txt')
N = 25
alloc = invest(f_in, w_in, N)
print('[Part B] Allocation:', alloc)
