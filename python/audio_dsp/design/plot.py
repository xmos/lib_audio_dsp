# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""
Helper functions for displaying plots in the jupyter notebook pipeline
design.
"""

import matplotlib.pyplot as plt
import numpy as np

import audio_dsp.dsp.utils as utils


def plot_frequency_response(f, h, name="", range=50):
    """
    Plot the frequency response.

    Parameters
    ----------
    f : numpy.ndarray
        Frequencies (The X axis)
    h : numpy.ndarray
        Frequency response at the corresponding frequencies in ``f``
    name : str
        String to include in the plot title, if not set there will be no title.
    range : int | float
        Set the Y axis lower limit in dB, upper limit will be the maximum
        magnitude.
    """
    h_db = utils.db(h)

    y_max = np.max(h_db) + 1
    y_min = y_max - range

    fig, axs = plt.subplots(2, 1, sharex=True)
    if name:
        fig.suptitle(f"{name} frequency response".title())
    axs[0].semilogx(f, h_db)
    axs[0].set_ylim([y_min, y_max])
    axs[0].set_xlim([20, 20000])
    axs[0].set_ylabel("Magnitude (dB)")
    axs[0].grid()

    axs[1].semilogx(f, np.angle(h))
    axs[1].set_ylabel("Phase (rad)")
    axs[1].grid()
    axs[1].set_xlabel("Frequency (Hz)")

    plt.show()

    return fig
