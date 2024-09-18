import numpy as np
import argparse
import math
import os

# TODO move to common code
def quant(coefs, exp):
    quantised = np.rint(np.ldexp(coefs, exp))
    quantised_and_clipped = np.clip(quantised, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    assert np.allclose(quantised, quantised_and_clipped)
    return np.array(quantised_and_clipped, dtype=np.int64)


def emit_filter(fd_block_coefs, name, file_handle, taps_per_block, bits_per_element = 32):

    assert len(fd_block_coefs.shape) == 2
    phases, fd_block_length = fd_block_coefs.shape

    # print('phases', phases, 'fd_block_length', fd_block_length)

    fd_block_length -= 1 #due to the way we pack the NQ and DC

    int32_elements = phases * fd_block_length * 2 #2 for complex

    coef_data_name = 'coefs_' + str(name)
    file_handle.write('int32_t __attribute__((aligned (8))) ' + coef_data_name + '[' + str(int32_elements) + '] = {\n')

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

        quantised_coefs = quant(flat_fd_block, exp)
        block_properties.append([offset, exp])

        for quantised_coef in quantised_coefs:
            file_handle.write('%12d'%(quantised_coef))
            if counter != len(quantised_coefs)*phases:
                file_handle.write(',\t')
            if counter % 4 == 0:
                file_handle.write('\n')
            counter += 1
        offset += len(flat_fd_block)
    file_handle.write('};\n')

    # now emit the bfp_complex_s32_t struct array
    coef_blocks_name = "coef_blocks_" + name
    file_handle.write("bfp_complex_s32_t " + coef_blocks_name +'['+ str(len(block_properties)) + '] = {\n')
    counter = 1
    for offset, exp in block_properties:
        file_handle.write("\t{.data = (complex_s32_t*)(" + coef_data_name + " + " + str(offset) + ")," + 
                          " .length = " + str(fd_block_length) + ", .exp = " + str(-exp) + ", .flags = 0, .hr = 0}")

        if counter != len(block_properties):
            file_handle.write(',\t')
        file_handle.write('\n')
        counter += 1
    file_handle.write("};\n")

    # then emit the fd_FIR_data_t struct
    file_handle.write("fd_FIR_filter_t fd_fir_filter_" + name + ' = {\n')
    file_handle.write('\t.coef_blocks = '+coef_blocks_name+',\n')
    file_handle.write('\t.td_block_length = ' + str(fd_block_length*2) +',\n')
    file_handle.write('\t.block_count = ' + str(phases) + ',\n')
    file_handle.write('\t.taps_per_block = ' + str(taps_per_block) + ',\n')
    file_handle.write("};\n")


# TODO put this in a library for tb_block_fir and this to share
# emit the debug filter coefs
def emit_debug_filter(fh, coefs, name):
    filter_length = len(coefs)
    
    max_val = np.max(np.abs(coefs))
    _, e = np.frexp(max_val)
    exp = 31 - e

    quantised_filter = np.int32(np.rint(np.ldexp(coefs, exp)))
    quantised_filter = np.clip(quantised_filter, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    v = np.where(quantised_filter>0, np.iinfo(np.int32).max, np.iinfo(np.int32).min)

    # Convert to pythons arb precision ints
    max_accu = sum([a * b for a, b in zip(quantised_filter.tolist(), v.tolist())])

    prod_shr = int(np.ceil(np.log2(max_accu / np.iinfo(np.int64).max)))
    if prod_shr < 0:
        prod_shr = 0

    accu_shr = exp - prod_shr
    coef_data_name = 'debug_' + name + '_filter_taps'
    fh.write('int32_t __attribute__((aligned (8))) ' + coef_data_name + '[' + str(filter_length) + '] = {\n')
    
    counter = 1
    for val in coefs:
        int_val = np.int32(np.rint(np.ldexp(val, exp)))
        fh.write('%12d'%(int_val))
        if counter != filter_length:
            fh.write(',\t')
        if counter % 4 == 0:
            fh.write('\n')
        counter += 1
    fh.write('};\n\n')

    struct_name = "td_block_debug_fir_filter_" + name

    fh.write("td_block_debug_fir_filter_t " + struct_name + ' = {\n')
    fh.write('\t.coefs = '+ coef_data_name +',\n')
    fh.write('\t.length = ' + str(filter_length) +',\n')
    fh.write('\t.exponent = ' + str(-exp) + ',\n')
    fh.write('\t.accu_shr = ' + str(accu_shr) + ',\n')
    fh.write('\t.prod_shr = ' + str(prod_shr) + ',\n')
    fh.write("};\n")
    fh.write("\n")

    return struct_name

def process_array(td_coefs, filter_name, output_path, 
                    frame_advance, frame_overlap, td_block_length, gain_dB = 0.0, debug = False):
    
    if not math.log2(td_block_length).is_integer():
        print("Error: td_block_length is not a power of two")
        exit(1)

    output_file_name = os.path.join(output_path, filter_name + '.h')

    original_td_filter_length = len(td_coefs)

    taps_per_block = td_block_length//2 + 1 - frame_overlap

    # the taps_per_block cannot exceed the frame_advance as this would result in
    # outputting the same samples multiple times.
    if taps_per_block > frame_advance:
        taps_per_block = frame_advance

    print('frame_advance', frame_advance)
    print('td_block_length', td_block_length)
    print('frame_overlap', frame_overlap)
    print('taps_per_block', taps_per_block)

    # Calc the length the filter need to be to fill all blocks when zero padded
    adjusted_td_length = ((original_td_filter_length + (taps_per_block - 1))//taps_per_block)*taps_per_block 

    print('adjusted_td_length', adjusted_td_length)
    # check length is efficient for td_block_length
    if original_td_filter_length % taps_per_block != 0:
        print("Warning: Chosen td_block_length and frame_overlap is not maximally efficient for filter of length", original_td_filter_length)
        print("         Better would be:", adjusted_td_length, 'taps, currently it will be padded with', adjusted_td_length-original_td_filter_length, 'zeros.')
    
    # pad filters
    phases = adjusted_td_length//taps_per_block
    assert adjusted_td_length%taps_per_block == 0
    print('phases', phases)

    if adjusted_td_length != original_td_filter_length:
        padding = np.zeros(adjusted_td_length - original_td_filter_length)
        prepared_coefs = np.concatenate((td_coefs, padding))
    else:
        prepared_coefs = td_coefs

    # Apply the gains
    prepared_coefs *= 10.**(gain_dB/20.)

    assert len(prepared_coefs)%taps_per_block == 0

    # split into blocks
    blocked = np.reshape(prepared_coefs, (-1, taps_per_block))
    print('blocked', blocked.shape)

    padding_per_block = td_block_length - taps_per_block
    print('padding_per_block', padding_per_block)

    # zero pad the filter taps
    blocked_and_padded = np.concatenate((blocked, np.zeros((phases, padding_per_block))), axis = 1)

    print('blocked_and_padded', blocked_and_padded.shape)
    # transform to the frequency domain
    Blocked_and_padded = np.fft.rfft(blocked_and_padded)

    print('Blocked_and_padded', Blocked_and_padded.shape)

    with open(output_file_name, 'w') as fh:
        fh.write('#include "dsp/fd_block_fir.h"\n\n')

        emit_filter(Blocked_and_padded, filter_name, fh, taps_per_block)

        if debug:
            emit_debug_filter(fh, coefs, filter_name)

            fh.write("#define debug_" + filter_name + "_DATA_BUFFER_ELEMENTS (" + str(len(coefs)) + ")\n")
            fh.write("\n")

        prev_buffer_length = td_block_length - frame_advance
        data_buffer_length = phases * td_block_length

        data_memory = '((sizeof(bfp_complex_s32_t) * ' + str(phases) +') / sizeof(int32_t))'
        data_memory += ' + ('  + str(frame_overlap) + ')'
        data_memory += ' + ('  + str(prev_buffer_length) + ')'
        data_memory += ' + ('  + str(data_buffer_length) + ')'

        # emit the data define
        fh.write("//This is the count of int32_t words to allocate for one data channel.\n")
        fh.write("//i.e. int32_t channel_data[" + filter_name + "_DATA_BUFFER_ELEMENTS] = \{0\};\n")
        fh.write("#define " + filter_name + "_DATA_BUFFER_ELEMENTS (" + str(data_memory) + ")\n\n")


        fh.write("#define " + filter_name + "_TD_BLOCK_LENGTH (" + str(td_block_length) + ")\n")
        fh.write("#define " + filter_name + "_BLOCK_COUNT (" + str(phases) + ")\n")
        fh.write("#define " + filter_name + "_FRAME_ADVANCE (" + str(frame_advance) + ")\n")
            

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('block_length', type=int,
                    help='Length of a block. Must be a power of 2.')
    parser.add_argument('filter', type=str,
                    help='path to the filter(numpy format)')
    parser.add_argument('--frame_advance', type=int, default = None,
                    help='The count of new samples from one update to the next. Assumed block_length//2 if not given.')
    
    parser.add_argument('--frame_overlap', type=int, default = None,
                    help=' TODO . Defaults to 0(LTI filtering).')
    parser.add_argument('--gain', type=float, default=0.0,
                    help='Apply a gain to the output(dB).')
    parser.add_argument('--output', type=str, default=".",
                    help='Output location.')
    parser.add_argument('--debug', action="store_true", default=False,
                    help='Enable debug output.')
    parser.add_argument('--name', type=str, default=None,
                    help='Name for the filter(override the default which is the filename)')
    
    args = parser.parse_args()

    if args.frame_advance == None:
        frame_advance = args.block_length//2
    if args.frame_overlap == None:
        frame_overlap = 0

    output_path = os.path.realpath(args.output)
    filter_path = os.path.realpath(args.filter)
    gain_dB = args.gain

    if os.path.exists(filter_path):
        coefs = np.load(filter_path)
    else:
        print("Error: cannot find ", filter_path )
        exit(1)

    if args.name != None:
        filter_name = args.name
    else:
        p = os.path.basename(filter_path)
        filter_name = p.split('.')[0]

    process_array(coefs, filter_name, output_path, frame_advance, 
                    frame_overlap, args.block_length, gain_dB = gain_dB, debug = args.debug)