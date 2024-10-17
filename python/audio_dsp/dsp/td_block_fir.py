import numpy as np
import argparse
import os
import ref_fir as rf

def calc_max_accu(quantised_coefs, VPU_shr = 30):
    v = np.where(quantised_coefs > 0, np.iinfo(np.int32).max, np.iinfo(np.int32).min)
    v = np.array(v, dtype=np.int64)
    accu = 0
    for x, y in zip(v, quantised_coefs):
        accu += np.int64(np.rint((x*y)/2**VPU_shr))
    return accu

def emit_filter(fh, coefs_padded, name, block_length, bits_per_element = 32):

    VPU_shr = 30 #the CPU shifts the product before accumulation
    VPU_accu_bits = 40

    # reverse the filter
    coefs_padded = coefs_padded[::-1]

    coef_data_name = 'coefs_' + name

    max_val = np.max(np.abs(coefs_padded))
    _, e = np.frexp(max_val)
    exp = bits_per_element - 2 - e

    quantised_coefs = rf.quant(coefs_padded, exp)
    max_accu = calc_max_accu(quantised_coefs, VPU_shr)

    # This guarentees no accu overflow
    while max_accu > 2**(VPU_accu_bits-1) - 1:
        exp -= 1
        quantised_coefs = rf.quant(coefs_padded, exp)
        max_accu = calc_max_accu(quantised_coefs)

    fh.write('int32_t __attribute__((aligned (8))) ' + coef_data_name + '[' + str(len(coefs_padded)) + '] = {\n')
    counter = 1
    for val in quantised_coefs:
        fh.write('%12d'%(val))
        if counter != len(coefs_padded):
            fh.write(',\t')
        if counter % 4 == 0:
            fh.write('\n')
        counter += 1
    fh.write('};\n')

    if VPU_shr-exp > 0:
        accu_shr = 0
        accu_shl = exp - VPU_shr
    else:
        accu_shr = exp - VPU_shr
        accu_shl = 0

    # then emit the td_block_fir_filter_t struct
    filter_struct_name = "td_block_fir_filter_" + name
    fh.write("td_block_fir_filter_t " + filter_struct_name + ' = {\n')
    fh.write('\t.coefs = '+ coef_data_name +',\n')
    fh.write('\t.block_count = ' + str(len(coefs_padded) // block_length) +',\n')
    fh.write('\t.accu_shr = ' + str(accu_shr) + ',\n')
    fh.write('\t.accu_shl = ' + str(accu_shl) + ',\n') 
    fh.write("};\n")
    fh.write("\n")
    
    return filter_struct_name

def process_array(coefs, filter_name, output_path, gain_dB = 0.0, debug = False, block_length = 8, silent = False):

    output_file_name = os.path.join(output_path, filter_name + '.h')

    original_filter_length = len(coefs)

    # this is the above but rounded up to the nearest block_length
    target_filter_bank_length = ((original_filter_length + block_length - 1)//block_length)*block_length

    if original_filter_length != target_filter_bank_length:
        if not silent: print("Warning: ", filter_name, " will be zero padded to length ", target_filter_bank_length)
        padding = np.zeros(target_filter_bank_length - original_filter_length)
        prepared_coefs = np.concatenate((coefs, padding))
    else:
        prepared_coefs = coefs
        
    # Apply the gains
    prepared_coefs *= 10.**(gain_dB/20.)

    with open(output_file_name, 'w') as fh:
        fh.write('#include "dsp/td_block_fir.h"\n\n')

        # The count of blocks in the filter ( the data is at least 2 more)
        filter_block_count = target_filter_bank_length // block_length

        emit_filter(fh, prepared_coefs, filter_name, block_length)

        if debug:
            rf.emit_debug_filter(fh, coefs, filter_name)

            fh.write("#define debug_" + filter_name + "_DATA_BUFFER_ELEMENTS (" + str(len(coefs)) + ")\n")
            fh.write("\n")

        # emit the data define
        data_block_count = filter_block_count + 2
        fh.write("//This is the count of int32_t words to allocate for one data channel.\n")
        fh.write("//i.e. int32_t channel_data[" + filter_name + "_DATA_BUFFER_ELEMENTS] = \{0\};\n")
        fh.write("#define " + filter_name + "_DATA_BUFFER_ELEMENTS (" + str(data_block_count*block_length) + ")\n\n")
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('filter', type=str,
                    help='path to the filter(numpy format)')
    parser.add_argument('--gain', type=float, default=0.0,
                    help='Apply a gain to the output(dB).')
    parser.add_argument('--output', type=str, default=".",
                    help='Output location.')
    parser.add_argument('--debug', action="store_true", default=False,
                    help='Enable debug output.')
    parser.add_argument('--name', type=str, default=None,
                    help='Name for the filter(override the default which is the filename)')
    
    args = parser.parse_args()

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

    process_array(coefs, filter_name, output_path, gain_dB, debug = args.debug)