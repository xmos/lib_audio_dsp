# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Frequency domain block FIR generator."""

import numpy as np
import argparse
import math
import os
import audio_dsp.dsp.ref_fir as rf


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

        quantised_coefs = rf.quant(flat_fd_block, exp)
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


def generate_fd_fir(
    td_coefs: np.ndarray,
    filter_name: str,
    output_path: str,
    frame_advance: int,
    frame_overlap: int,
    td_block_length: int,
    gain_dB=0.0,
    verbose=False,
    warn=False,
    error=True,
    debug=False,
    ):
    """
    Convert the input array into a header to be included in a C project.

    Parameters
    ----------
    td_coefs : np.ndarray
        This is a 1D numpy float array of the coefficients of the filter.
    filter_name : str
        For use in identification of the filter from within the C code. All structs and defiens that pertain to this filter will contain this identifier.
    output_path : str
        Where to output the resultinng header file.
    frame_advance : int
        The numer of samples etween susequent frames.
    frame_overlap : int
        When the convolution is performed it will always output frame_advance samples plus an optional frame_overlap.
    td_block_length : int
        The size in samples of a frame, measured in time domain samples.
    gain_dB : float, optional
        A gain applied to the filters output, by default 0.0
    debug : bool, optional
        If enabled then this will emit a debug struct, by default False
    warn : bool, optional
        Enable to emit warnings, by default False
    error : bool, optional
        Enable to emit error fix suggestions, by default True
    verbose : bool, optional
        Enable verbose printinng, by default False

    Raises
    ------
        ValueError: Bad config - Should be fixed
        ValueError: Unachievable config - MUST be fixed
    """
    td_coefs = np.array(td_coefs, dtype=np.float64)

    if not math.log2(td_block_length).is_integer():
        if error:
            print("Error: td_block_length is not a power of two")
        raise ValueError("Bad config")

    output_file_name = os.path.join(output_path, filter_name + ".h")

    original_td_filter_length = len(td_coefs)
    if frame_advance < 1:
        if error:
            print("Error: cannot have a zero or negative frame_advance")
        raise ValueError("Bad config")

    if frame_overlap == None:
        frame_overlap = 0

    if frame_overlap < 0:
        if error:
            print("Error: cannot have a negative frame_overlap")
        raise ValueError("Bad config")

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

    if minimum_output_samples <= td_block_length + 1 - original_td_filter_length:
        # This is a single-phase FIR
        if verbose:
            print("This is a single-phase FIR")

        taps_per_phase = original_td_filter_length
        actual_output_sample_count = td_block_length + 1 - original_td_filter_length

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
        actual_output_sample_count = td_block_length + 1 - taps_per_phase

        if actual_output_sample_count < minimum_output_samples:
            achievable_frame_overlap = actual_output_sample_count - frame_advance
            achievable_frame_advance = actual_output_sample_count - frame_overlap
            achievable_block_length = minimum_output_samples - 1 + original_td_filter_length
            if error:
                print("Error: frame_overlap of", frame_overlap, "is unachievable.")
                print("\tOption 1: reduce frame_overlap to", achievable_frame_overlap)
                print("\tOption 2: decrease the frame_advance to", achievable_frame_advance)
                print("\tOption 3: increase the td_block_length to", achievable_block_length)
            raise ValueError("Unachievable config")

        phases = (original_td_filter_length + taps_per_phase - 1) // taps_per_phase

        assert phases > 1

        new_frame_overlap = td_block_length + 1 - taps_per_phase - frame_advance

    if new_frame_overlap != frame_overlap:
        if warn:
            print(
                "Warning: requested a frame overlap of",
                frame_overlap,
                "but will get ",
                new_frame_overlap,
            )
            print(
                "To increase efficiency, try increasing the length of the filter by",
                (new_frame_overlap - frame_overlap) * phases,
            )
        assert new_frame_overlap > frame_overlap
        frame_overlap = new_frame_overlap

    if verbose:
        print("actual_output_sample_count", actual_output_sample_count)
        print("frame_advance", frame_advance)
        print("td_block_length", td_block_length)
        print("frame_overlap", frame_overlap)
        print("taps_per_phase", taps_per_phase)
        print("phases", phases)

    # Calc the length the filter need to be to fill all blocks when zero padded
    adjusted_td_length = taps_per_phase * phases
    if verbose:
        print("adjusted_td_length", adjusted_td_length)
    assert adjusted_td_length >= original_td_filter_length

    # check length is efficient for td_block_length
    if original_td_filter_length % taps_per_phase != 0:
        if warn:
            print(
                "Warning: Chosen td_block_length and frame_overlap is not maximally efficient for filter of length",
                original_td_filter_length,
            )
            print(
                "         Better would be:",
                adjusted_td_length,
                "taps, currently it will be padded with",
                adjusted_td_length - original_td_filter_length,
                "zeros.",
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
    prepared_coefs *= 10.0 ** (gain_dB / 20.0)

    assert len(prepared_coefs) % taps_per_phase == 0

    # split into blocks
    blocked = np.reshape(prepared_coefs, (-1, taps_per_phase))

    padding_per_block = td_block_length - taps_per_phase

    # zero pad the filter taps
    blocked_and_padded = np.concatenate((blocked, np.zeros((phases, padding_per_block))), axis=1)

    # transform to the frequency domain
    Blocked_and_padded = np.fft.rfft(blocked_and_padded)

    with open(output_file_name, "w") as fh:
        fh.write('#include "dsp/fd_block_fir.h"\n\n')

        _emit_filter(Blocked_and_padded, filter_name, fh, taps_per_phase)

        if debug:
            rf.emit_debug_filter(fh, td_coefs, filter_name)

            fh.write(
                "#define debug_"
                + filter_name
                + "_DATA_BUFFER_ELEMENTS ("
                + str(len(td_coefs))
                + ")\n"
            )
            fh.write("\n")

        prev_buffer_length = td_block_length - frame_advance
        data_buffer_length = phases * td_block_length

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

        fh.write("#define " + filter_name + "_TD_BLOCK_LENGTH (" + str(td_block_length) + ")\n")
        fh.write("#define " + filter_name + "_BLOCK_COUNT (" + str(phases) + ")\n")
        fh.write("#define " + filter_name + "_FRAME_ADVANCE (" + str(frame_advance) + ")\n")
        fh.write("#define " + filter_name + "_FRAME_OVERLAP (" + str(frame_overlap) + ")\n")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional app description")

    parser.add_argument("block_length", type=int, help="Length of a block. Must be a power of 2.")
    parser.add_argument("filter", type=str, help="path to the filter(numpy format)")
    parser.add_argument(
        "--frame_advance",
        type=int,
        default=None,
        help="The count of new samples from one update to the next. Assumed block_length//2 if not given.",
    )

    parser.add_argument(
        "--frame_overlap", type=int, default=None, help=" TODO . Defaults to 0(LTI filtering)."
    )
    parser.add_argument("--gain", type=float, default=0.0, help="Apply a gain to the output(dB).")
    parser.add_argument("--output", type=str, default=".", help="Output location.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug output.")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name for the filter(override the default which is the filename)",
    )

    args = parser.parse_args()

    if args.frame_advance == None:
        frame_advance = args.block_length // 2
    else:
        frame_advance = args.frame_advance

    output_path = os.path.realpath(args.output)
    filter_path = os.path.realpath(args.filter)
    gain_dB = args.gain

    if os.path.exists(filter_path):
        coefs = np.load(filter_path)
    else:
        print("Error: cannot find ", filter_path)
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
        frame_advance,
        args.frame_overlap,
        args.block_length,
        gain_dB=gain_dB,
        debug=args.debug,
    )
