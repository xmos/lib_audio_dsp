import numpy as np
from scipy import signal 

def lowpass_filters(length, count):
    for i in range(count):
        t = signal.firwin(length, (i + 1) / (count + 2))
        name = 'lowpass_' + str(i)
        np.save(name, t)

def make_trivial(name):
    t = np.ones(1)
    np.save(name, t)

def make_random(name, length):
    t = np.random.uniform(-1, 1, length)
    np.save(name, t)

if __name__ == '__main__':
    lowpass_filters(4096, 16)

    make_random('random_63', 63)
    make_random('random_64', 4096)

    make_trivial('trivial')