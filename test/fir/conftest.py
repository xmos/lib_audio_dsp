import numpy as np
import scipy.signal as spsig
from pathlib import Path

def pytest_sessionstart():

    gen_dir = Path(__file__).parent / "autogen"
    gen_dir.mkdir(exist_ok=True, parents=True)

    coeffs = np.zeros(1000)
    coeffs[0] = 1
    np.savetxt(Path(gen_dir, "passthrough_filter.txt"), coeffs)

    coeffs = np.arange(10, 0, -1)
    np.savetxt(Path(gen_dir, "descending_coeffs.txt"), coeffs)

    coeffs = spsig.firwin2(512, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    np.savetxt(Path(gen_dir, "simple_low_pass.txt"), coeffs)

    print("made things")
