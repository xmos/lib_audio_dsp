# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Common code for time and frequency domain block FIR generator."""

import numpy as np
import io
import os

# emit the debug filter coefs
def emit_debug_filter(fh: io.TextIOWrapper, coefs: np.ndarray, name: str):
    """
    Emit a debug section describing the filter to the header.

    Parameters
    ----------
    fh : io.TextIOWrapper
        File handle of the header to write to.
    coefs : np.ndarray
        Array of floats describing the filter.
    name : str
        Name of the filter.

    Returns
    -------
    str
        Name of the structure contining the deubg info.

    """
    filter_length = len(coefs)

    max_val = np.max(np.abs(coefs))
    _, e = np.frexp(max_val)
    exp = 31 - e

    quantised_filter = np.array(np.rint(np.ldexp(coefs, exp)), dtype=np.int32)
    quantised_filter = np.clip(quantised_filter, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    v = np.where(quantised_filter > 0, np.iinfo(np.int32).max, np.iinfo(np.int32).min)

    # Convert to pythons arb precision ints
    max_accu = sum([a * b for a, b in zip(quantised_filter.tolist(), v.tolist())])

    prod_shr = int(np.ceil(np.log2(max_accu / np.iinfo(np.int64).max)))
    if prod_shr < 0:
        prod_shr = 0

    accu_shr = exp - prod_shr
    coef_data_name = "debug_" + name + "_filter_taps"
    fh.write(
        "int32_t __attribute__((aligned (8))) "
        + coef_data_name
        + "["
        + str(filter_length)
        + "] = {\n"
    )

    counter = 1
    for val in coefs:
        int_val = np.int32(np.rint(np.ldexp(val, exp)))
        fh.write("%12d" % (int_val))
        if counter != filter_length:
            fh.write(",\t")
        if counter % 4 == 0:
            fh.write("\n")
        counter += 1
    fh.write("};\n\n")

    struct_name = "td_block_debug_fir_filter_" + name

    fh.write('#include "ref_fir.h"\n')
    fh.write("td_reference_fir_filter_t " + struct_name + " = {\n")
    fh.write("\t.coefs = " + coef_data_name + ",\n")
    fh.write("\t.length = " + str(filter_length) + ",\n")
    fh.write("\t.exponent = " + str(-exp) + ",\n")
    fh.write("\t.accu_shr = " + str(accu_shr) + ",\n")
    fh.write("\t.prod_shr = " + str(prod_shr) + ",\n")
    fh.write("};\n")
    fh.write("\n")

    return struct_name


def generate_debug_fir(
    td_coefs: np.ndarray,
    filter_name: str,
    output_path: str,
    frame_advance=None,
    frame_overlap=None,
    td_block_length=None,
    gain_db=0.0,
    verbose=False,
):
    """Convert the input array into a header to be included in a C debug tests."""
    output_file_name = os.path.join(output_path, filter_name + "_debug.h")
    td_coefs = np.array(td_coefs, dtype=np.float64)

    with open(output_file_name, "w") as fh:
        fh.write('#include "dsp/fd_block_fir.h"\n\n')

        emit_debug_filter(fh, td_coefs, filter_name)

        fh.write(
            "#define debug_" + filter_name + "_DATA_BUFFER_ELEMENTS (" + str(len(td_coefs)) + ")\n"
        )
        fh.write("\n")
