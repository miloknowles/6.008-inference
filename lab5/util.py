from __future__ import division

import numpy as np

class DefaultDict(dict):
    """
    Like an ordinary dictionary, but returns the result of calling
    default_factory when the key is missing.

    For example, a counter (see above) could be implemented as either
    my_counter = Counter()
    my_counter = DefaultDict(lambda : 0)

    This is modeled after the collections.defaultdict class, which is 
    only available in Python 2.7+.
    """

    def __init__(self, default_factory):
        """
        default_factory is a function that takes no arguments and
        returns the default value
        """
        self._default_factory = default_factory

    def __missing__(self, key):
        self[key] = self._default_factory()
        return self[key]

#-----------------------------------------------------------------------------
# A general purpose Distribution class for finite discrete random variables
#

class Distribution(dict):
    """
    The Distribution class extends the Python dictionary such that
    each key's value should correspond to the probability of the key.

    For example, here's how you can create a random variable X that takes on
    value 'spam' with probability .7 and 'eggs' with probability .3:

    X = Distribution()
    X['spam'] = .7
    X['eggs'] = .3

    Methods
    -------
    renormalize():
      scales all the probabilities so that they sum to 1
    get_mode():
      returns an item with the highest probability, breaking ties arbitrarily
    sample():
      draws a sample from the Distribution
    """
    def __missing__(self, key):
        # if the key is missing, return probability 0
        return 0

    def renormalize(self):
        normalization_constant = sum(self.values())
        assert normalization_constant > 0, "Probabilities shouldn't all be 0"
        for key in self.keys():
            self[key] /= normalization_constant

    def get_mode(self):
        maximum = -1
        arg_max = None

        for key in self.keys():
            if self[key] > maximum:
                arg_max = key
                maximum = self[key]

        return arg_max

    def sample(self):
        keys  = []
        probs = []
        for key, prob in self.items():
            keys.append(key)
            probs.append(prob)

        rand_idx = np.where(np.random.multinomial(1, probs))[0][0]
        return keys[rand_idx]




