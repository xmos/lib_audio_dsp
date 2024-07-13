import math
import numpy as np

def gen_wn():
    MEAN = 0
    STD = 1
    N_SAMPS = 1024

    output_samps = []

    gen_rand = np.random.normal(MEAN, STD, size=N_SAMPS)
    gen_rand /= max(gen_rand)
    for i in range(N_SAMPS):
        y = (-1 if gen_rand[i] < -1 else (1 if gen_rand[i] > 1 else gen_rand[i]))
        y_q31 = int(y * ((2 ** 31)- 1))
        output_samps.append(y_q31)

    with open("out.txt", "w") as f:
        for samp in output_samps:
            f.write(f"{samp},")
        


if __name__ == "__main__":
    gen_wn()
