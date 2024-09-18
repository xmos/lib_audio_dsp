import numpy as np
from scipy import signal 

def bandpass_filters(length, count):
    for i in range(count):
        t = signal.firwin(length, (i + 1) / (count + 2))
        name = 'test_' + str(i)
        np.save(name, t)

if __name__ == '__main__':
    bandpass_filters(4096, 1)