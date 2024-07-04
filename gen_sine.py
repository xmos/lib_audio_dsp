import math

def gen_sine():
    FREQ = 1000  # Hertz
    SAMPLE_RATE = 48000  # Hertz
    PERIODS = 1

    num_samps = SAMPLE_RATE // (FREQ * PERIODS)
    output_samps = []
    for i in range(num_samps):
        angle = (2 * math.pi * i) / num_samps
        y = math.sin(angle)
        y = (-1 if y < -1 else (1 if y > 1 else y))
        y_q31 = int(y * ((2 ** 31)- 1))
        output_samps.append(y_q31)

    with open("out.txt", "w") as f:
        for samp in output_samps:
            f.write(f"{samp},")
        


if __name__ == "__main__":
    gen_sine()
