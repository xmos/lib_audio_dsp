# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Frequency domain block FIR generator."""

import numpy as np
import argparse
import math
import os
from pathlib import Path
from audio_dsp.dsp import generic as dspg
import audio_dsp.dsp.utils as utils
import warnings
from copy import deepcopy
from typing import Optional, Tuple


class fir_block_fd(dspg.dsp_block):
    """
    An FIR filter, implemented in block form in the frequency domain.

    This will also autogenerate a .c and .h file containing the
    optimised block filter structures, designed for use in C.

    Parameters
    ----------
    coeffs_path : Path
        Path to a file containing the coefficients, in a format supported by
        `np.loadtxt <https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html>`_.
    filter_name : str
        For use in identification of the filter from within the C code.
        All structs and defines that pertain to this filter will contain
        this identifier.
    output_path : str
        Where to output the resulting header file.
    frame_advance : int
        The number of new samples between subsequent frames.
    frame_overlap : int, optional
        The number of additional samples to output per frame. This allows
        windowing between frames to occur. By default no overlap occurs.
    nfft : int, optional
        The FFT size in samples of a frame, measured in time domain samples.
        If this is not set, the FFT size is set automatically. An initial
        attempt of ``nfft = 2**(ceil(log2(frame_advance)) + 1)`` is made,
        but may need to be increased for longer overlaps. If it is set,
        it must be a power of 2.
    gain_db : float, optional
        A gain applied to the filters output, by default 0.0

    Attributes
    ----------
    coeffs : np.ndarray
        Time domain coefficients
    n_taps : int
        Length of time domain filter
    frame_advance : int
        The number of new samples between subsequent frames.
    frame_overlap : int
        The number of additional samples to output per frame.
    nfft : int, optional
        The FFT size in samples of a frame.
    coeffs_fs : np.ndarray
        The frequency domain coefficients.
    n_fd_buffers : int
        The number of frames of frequency domain coefficients, set the
        number of buffers that need to be saved.
    td_buffer : np.ndarray
        Buffer of last nfft time domain inputs in floating point format
    td_buffer_int : list
        Buffer of last nfft time domain inputs in fixed point format
    fd_buffer : np.ndarray
        Buffer of last n_fd_buffers of the spectrums of previous
        td_buffers.

    """

    def __init__(
        self,
        fs: float,
        n_chans: int,
        coeffs_path: Path,
        filter_name: str,
        output_path: Path,
        frame_advance: int,
        frame_overlap: int = 0,
        nfft: Optional[int] = None,
        gain_db: float = 0.0,
        Q_sig: int = dspg.Q_SIG,
    ):
        super().__init__(fs, n_chans, Q_sig)
        self.coeffs = np.loadtxt(coeffs_path)
        self.n_taps = len(self.coeffs)

        self.frame_advance = frame_advance

        filter_struct_name, self.coeffs_fd, quantized_coefs, self.taps_per_phase = generate_fd_fir(
            self.coeffs,
            filter_name,
            output_path,
            frame_advance,
            frame_overlap,
            nfft,
            gain_db=gain_db,
        )

        self.nfft = 2 * (self.coeffs_fd.shape[1] - 1)
        self.n_fd_buffers = self.coeffs_fd.shape[0]

        self.reset_state()

    def reset_state(self) -> None:
        """Reset all the delay line values to zero."""
        # buffer_len = (self.n_taps + self.block_len - 1)
        self.td_buffer = np.zeros((self.n_chans, self.nfft))
        self.td_buffer_int = [[0] * self.nfft for _ in range(self.n_chans)]

        self.fd_buffer = np.zeros(
            (self.n_chans, self.n_fd_buffers, self.nfft // 2 + 1), dtype=np.complex128
        )
        return

    def process_frame(self, frame: list):
        """Update the buffer with the current samples and convolve with
        the filter coefficients, using floating point math.

        Parameters
        ----------
        frame : list[float]
            The input samples to be processed.

        Returns
        -------
        float
            The processed output sample.
        """
        n_outputs = len(frame)
        frame_size = frame[0].shape[0]
        output = deepcopy(frame)
        for chan in range(n_outputs):
            self.td_buffer[chan, -self.frame_advance :] = frame[chan]
            self.fd_buffer[chan, 0, :] = np.fft.rfft(self.td_buffer[chan])

            output_spect = np.sum(self.fd_buffer[chan] * self.coeffs_fd, axis=(0))
            output_sig = np.fft.irfft(output_spect)
            output[chan] = output_sig[-self.frame_advance :]

            self.td_buffer[chan] = np.roll(self.td_buffer[chan], -self.frame_advance)
            self.fd_buffer[chan] = np.roll(self.fd_buffer[chan], 1, axis=0)

        return output


def _emit_filter(fd_block_coefs, name, file_handle, taps_per_block, bits_per_element=32):
    assert len(fd_block_coefs.shape) == 2
    phases, fd_block_length = fd_block_coefs.shape

    # print('phases', phases, 'fd_block_length', fd_block_length)

    fd_block_length -= 1  # due to the way we pack the NQ and DC

    int32_elements = phases * fd_block_length * 2  # 2 for complex

    coef_data_name = "coefs_" + str(name)
    file_handle.write(
        "int32_t __attribute__((aligned (8))) "
        + coef_data_name
        + "["
        + str(int32_elements)
        + "] = {\n"
    )

    block_properties = []

    # block_length = len(data[0])//2 #divide by two to make it complex
    offset = 0
    counter = 1

    for fd_block in fd_block_coefs:
        assert len(fd_block) > 1

        # flatten the real and imag
        flat_fd_block = np.hstack(tuple(zip(fd_block.real, fd_block.imag)))

        # move the NQ (to where the xcore expects it to be)
        flat_fd_block[1] = flat_fd_block[-2]

        # trim off the end (unused data)
        flat_fd_block = flat_fd_block[:-2]

        # calc the exponent
        _, exponents = np.frexp(flat_fd_block)
        e = max(exponents)
        exp = bits_per_element - e - 1

        quantised_coefs = utils.quantize_array(flat_fd_block, exp)
        block_properties.append([offset, exp])

        for quantised_coef in quantised_coefs:
            file_handle.write("%12d" % (quantised_coef))
            if counter != len(quantised_coefs) * phases:
                file_handle.write(",\t")
            if counter % 4 == 0:
                file_handle.write("\n")
            counter += 1
        offset += len(flat_fd_block)
    file_handle.write("};\n")

    # now emit the bfp_complex_s32_t struct array
    coef_blocks_name = "coef_blocks_" + name
    file_handle.write(
        "bfp_complex_s32_t " + coef_blocks_name + "[" + str(len(block_properties)) + "] = {\n"
    )
    counter = 1
    for offset, exp in block_properties:
        file_handle.write(
            "\t{.data = (complex_s32_t*)("
            + coef_data_name
            + " + "
            + str(offset)
            + "),"
            + " .length = "
            + str(fd_block_length)
            + ", .exp = "
            + str(-exp)
            + ", .flags = 0, .hr = 0}"
        )

        if counter != len(block_properties):
            file_handle.write(",\t")
        file_handle.write("\n")
        counter += 1
    file_handle.write("};\n")

    # then emit the fd_fir_data_t struct
    file_handle.write("fd_fir_filter_t fd_fir_filter_" + name + " = {\n")
    file_handle.write("\t.coef_blocks = " + coef_blocks_name + ",\n")
    file_handle.write("\t.td_block_length = " + str(fd_block_length * 2) + ",\n")
    file_handle.write("\t.block_count = " + str(phases) + ",\n")
    file_handle.write("\t.taps_per_block = " + str(taps_per_block) + ",\n")
    file_handle.write("};\n")

    return name, quantised_coefs


def _get_filter_phases(
    nfft,
    original_td_filter_length,
    frame_overlap,
    frame_advance,
    auto_block_length,
    verbose,
) -> Tuple[int, int, int, int, int]:
    """Calculate the number of phases of the filter, and check it will work.

    If using auto_block_length, increase the block length and recursively
    recall this function until the filter works.
    """
    # for every frame advance we must output at least frame_advance samples plus the requested frame_overlap samples
    minimum_output_samples = frame_overlap + frame_advance

    if minimum_output_samples <= nfft + 1 - original_td_filter_length:
        # This is a single-phase FIR
        if verbose:
            print("This is a single-phase FIR")

        taps_per_phase = original_td_filter_length
        actual_output_sample_count = nfft + 1 - original_td_filter_length

        phases = 1

        # update the frame_overlap
        new_frame_overlap = actual_output_sample_count - frame_advance

    else:
        # This is a multi-phase FIR
        if verbose:
            print("This is a multi-phase FIR")

        # want to work out the minimum number of phases that provides the required
        # (frame_overlap+frame_advance) output samples, with the constraint of the
        # taps_per_phase must be no greater than frame_advance.

        # these must be true for a multi-phase implementation
        taps_per_phase = frame_advance
        actual_output_sample_count = nfft + 1 - taps_per_phase

        if actual_output_sample_count < minimum_output_samples:
            if auto_block_length:
                print(f"Auto block length, trying next size up, was: {nfft}, now: {nfft * 2}")
                # increase block length to get enough output samples
                nfft *= 2
                # recursion in case we can now do a single phase filter
                return _get_filter_phases(
                    nfft,
                    original_td_filter_length,
                    frame_overlap,
                    frame_advance,
                    auto_block_length,
                    verbose,
                )

            else:
                achievable_frame_overlap = actual_output_sample_count - frame_advance
                achievable_frame_advance = actual_output_sample_count - frame_overlap
                achievable_block_length = minimum_output_samples - 1 + original_td_filter_length
                if verbose:
                    print("Error")
                    print("\tOption 1: reduce frame_overlap to", achievable_frame_overlap)
                    print("\tOption 2: decrease the frame_advance to", achievable_frame_advance)
                    print("\tOption 3: increase the td_block_length to", achievable_block_length)
                raise ValueError("Bad config: frame_overlap of", frame_overlap, "is unachievable.")

        phases = (original_td_filter_length + taps_per_phase - 1) // taps_per_phase

        assert phases > 1

        new_frame_overlap = nfft + 1 - taps_per_phase - frame_advance

    return nfft, phases, taps_per_phase, new_frame_overlap, actual_output_sample_count


def generate_fd_fir(
    td_coefs: np.ndarray,
    filter_name: str,
    output_path: Path,
    frame_advance: int,
    frame_overlap: int = 0,
    nfft: Optional[int] = None,
    gain_db: float = 0.0,
    verbose=False,
):
    """
    Convert the input filter coefficients array into a header with block
    frequency domain structures to be included in a C project.

    Parameters
    ----------
    td_coefs : np.ndarray
        This is a 1D numpy float array of the coefficients of the filter.
    filter_name : str
        For use in identification of the filter from within the C code.
        All structs and defines that pertain to this filter will contain
        this identifier.
    output_path : str
        Where to output the resulting header file.
    frame_advance : int
        The number of new samples between subsequent frames.
    frame_overlap : int, optional
        The number of additional samples to output per frame. This allows
        windowing between frames to occur. By default no overlap occurs.
    nfft : int, optional
        The FFT size in samples of a frame, measured in time domain samples.
        If this is not set, the FFT size is set automatically. An initial
        attempt of ``nfft = 2**(ceil(log2(frame_advance)) + 1)`` is made,
        but may need to be increased for longer overlaps. If it is set,
        it must be a power of 2.
    gain_db : float, optional
        A gain applied to the filters output, by default 0.0
    verbose : bool, optional
        Enable verbose printing, by default False

    Raises
    ------
        ValueError: Bad config - Must be fixed
    """
    td_coefs = np.array(td_coefs, dtype=np.float64)

    if frame_advance < 64:
        warnings.warn(
            "For frame_advance < 64, a time domain implementation is likely more"
            "efficient, please see AN02027, and try generate_td_fir instead.",
            UserWarning,
        )

    if not nfft:
        auto_block_length = True
        nfft = 2 ** (np.ceil(np.log2(frame_advance)).astype(int) + 1)
    elif not math.log2(nfft).is_integer():
        raise ValueError("Bad config: nfft is not a power of two")
    else:
        auto_block_length = False

    output_file_name = os.path.join(output_path, filter_name + ".h")

    original_td_filter_length = len(td_coefs)
    if frame_advance < 1:
        raise ValueError("Bad config: cannot have a zero or negative frame_advance")

    if frame_overlap == None:
        frame_overlap = 0

    if frame_overlap < 0:
        raise ValueError("Bad config: cannot have a negative frame_overlap")

    # for every frame advance we must output at least frame_advance samples plus the requested frame_overlap samples
    minimum_output_samples = frame_overlap + frame_advance

    if verbose:
        print(
            "original_td_filter_length:",
            original_td_filter_length,
            "frame_overlap",
            frame_overlap,
            "minimum_output_samples",
            minimum_output_samples,
        )

    nfft, phases, taps_per_phase, new_frame_overlap, actual_output_sample_count = (
        _get_filter_phases(
            nfft,
            original_td_filter_length,
            frame_overlap,
            frame_advance,
            auto_block_length,
            verbose,
        )
    )

    if new_frame_overlap != frame_overlap:
        warnings.warn(
            f"Requested a frame overlap of {frame_overlap}, but will get"
            f" {new_frame_overlap}. \nTo increase efficiency, try increasing the length of the filter"
            f" by {(new_frame_overlap - frame_overlap) * phases}.",
            UserWarning,
        )
        assert new_frame_overlap > frame_overlap
        frame_overlap = new_frame_overlap

    if verbose:
        print("actual_output_sample_count", actual_output_sample_count)
        print("frame_advance", frame_advance)
        print("nfft", nfft)
        print("frame_overlap", frame_overlap)
        print("taps_per_phase", taps_per_phase)
        print("phases", phases)

    # Calc the length the filter need to be to fill all blocks when zero padded
    adjusted_td_length = taps_per_phase * phases
    if verbose:
        print("adjusted_td_length", adjusted_td_length)
    assert adjusted_td_length >= original_td_filter_length

    # check length is efficient for nfft
    if original_td_filter_length % taps_per_phase != 0:
        warnings.warn(
            f"Chosen nfft and frame_overlap is not maximally "
            f"efficient for filter of length {original_td_filter_length}.\n"
            f"Better would be: {adjusted_td_length} taps, currently it will be padded with "
            f"{adjusted_td_length - original_td_filter_length} zeros.",
            UserWarning,
        )

    # pad filters
    assert adjusted_td_length % taps_per_phase == 0

    # if phases is 1 then we can have an extra tap

    if adjusted_td_length != original_td_filter_length:
        padding = np.zeros(adjusted_td_length - original_td_filter_length)
        prepared_coefs = np.concatenate((td_coefs, padding))
    else:
        prepared_coefs = td_coefs

    # Apply the gains
    prepared_coefs *= 10.0 ** (gain_db / 20.0)

    assert len(prepared_coefs) % taps_per_phase == 0

    # split into blocks
    blocked = np.reshape(prepared_coefs, (-1, taps_per_phase))

    padding_per_block = nfft - taps_per_phase

    # zero pad the filter taps
    blocked_and_padded = np.concatenate((blocked, np.zeros((phases, padding_per_block))), axis=1)

    # transform to the frequency domain
    coeffs_fd = np.fft.rfft(blocked_and_padded)

    with open(output_file_name, "w") as fh:
        fh.write('#include "dsp/fd_block_fir.h"\n\n')

        filter_struct_name, quantized_coefs = _emit_filter(
            coeffs_fd, filter_name, fh, taps_per_phase
        )

        prev_buffer_length = nfft - frame_advance
        data_buffer_length = phases * nfft

        data_memory = "((sizeof(bfp_complex_s32_t) * " + str(phases) + ") / sizeof(int32_t))"
        data_memory += " + (" + str(frame_overlap) + ")"
        data_memory += " + (" + str(prev_buffer_length) + ")"
        data_memory += " + (" + str(data_buffer_length) + ")"
        data_memory += " + 2"

        # emit the data define
        fh.write("//This is the count of int32_t words to allocate for one data channel.\n")
        fh.write(
            "//i.e. int32_t channel_data[" + filter_name + "_DATA_BUFFER_ELEMENTS] = { 0 };\n"
        )
        fh.write("#define " + filter_name + "_DATA_BUFFER_ELEMENTS (" + str(data_memory) + ")\n\n")

        fh.write("#define " + filter_name + "_TD_BLOCK_LENGTH (" + str(nfft) + ")\n")
        fh.write("#define " + filter_name + "_BLOCK_COUNT (" + str(phases) + ")\n")
        fh.write("#define " + filter_name + "_FRAME_ADVANCE (" + str(frame_advance) + ")\n")
        fh.write("#define " + filter_name + "_FRAME_OVERLAP (" + str(frame_overlap) + ")\n")

    return filter_struct_name, coeffs_fd, quantized_coefs, taps_per_phase


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional app description")

    parser.add_argument("filter", type=str, help="path to the filter (numpy format)")
    parser.add_argument(
        "frame_advance",
        type=int,
        help="The count of new samples from one update to the next. ",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name for the filter for use in identification of the filter from within the C code.",
    )

    parser.add_argument("--output", type=str, default=".", help="Output location.")
    parser.add_argument(
        "--frame_overlap",
        type=int,
        default=0,
        help="The number of additional samples to output per frame. This allows"
        "windowing between frames to occur. By default no overlap occurs.",
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=None,
        help="The FFT size in samples of a frame, measured in time domain "
        "samples. Must be a power of 2.",
    )

    parser.add_argument("--gain", type=float, default=0.0, help="Apply a gain to the output(dB).")

    args = parser.parse_args()

    output_path = os.path.realpath(args.output)
    filter_path = os.path.realpath(args.filter)
    gain_db = args.gain

    if os.path.exists(filter_path):
        coefs = np.load(filter_path)
    else:
        raise FileNotFoundError(f"Error: cannot find {filter_path}")
        exit(1)

    if args.name != None:
        filter_name = args.name
    else:
        p = os.path.basename(filter_path)
        filter_name = p.split(".")[0]

    generate_fd_fir(
        coefs,
        filter_name,
        output_path,
        args.frame_advance,
        frame_overlap=args.frame_overlap,
        nfft=args.nfft,
        gain_db=gain_db,
    )
