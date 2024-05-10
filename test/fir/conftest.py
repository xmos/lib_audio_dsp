import numpy as np
import scipy.signal as spsig
from pathlib import Path

def pytest_sessionstart():

    gen_dir = Path(__file__).parent / "autogen"
    gen_dir.mkdir(exist_ok=True, parents=True)

    coeffs = np.zeros(1000)
    coeffs[0] = 1
    np.savetxt(Path(gen_dir, "passthrough_filter.txt"), coeffs)

    coeffs = np.arange(10, 0, -1)/10
    np.savetxt(Path(gen_dir, "descending_coeffs.txt"), coeffs)

    coeffs = spsig.firwin2(512, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    np.savetxt(Path(gen_dir, "simple_low_pass.txt"), coeffs)

    coeffs = spsig.firwin2(32768, [0.0, 20/48000, 1.0], [0.0, 1.0, 1.0], antisymmetric=True)
    np.savetxt(Path(gen_dir, "aggressive_high_pass.txt"), coeffs)

    coeffs = spsig.firwin2(32767, [0.0, 0.5, 1.0], [0.5, 1.0, 2.0])
    np.savetxt(Path(gen_dir, "tilt.txt"), coeffs)

    coeffs = np.zeros(10000)
    coeffs[::8] = 1
    np.savetxt(Path(gen_dir, "comb.txt"), coeffs)


if __name__ == "__main__":
    pytest_sessionstart()