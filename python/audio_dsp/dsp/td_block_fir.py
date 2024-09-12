import numpy as np
import argparse
import os

def emit_filter(fh, coefs_padded, name, block_length, bits_per_element = 32):

    VPU_shr = 30 #the CPU shifts the product before accumulation
    VPU_accu_bits = 40

    # reverse the filter
    coefs_padded = coefs_padded[::-1]

    coef_data_name = 'coefs_' + name

    max_float_accu = np.sum(np.abs(coefs_padded))
    gain_dB = 20.*np.log10(max_float_accu)

    print("Given filter has a max gain of", str(np.round(gain_dB, 2)), "dB")
    if max_float_accu > 1.0:
        print("Warning: given filter's max gain exceeds the accumulators capacity")


    max_val = np.max(np.abs(coefs_padded))
    _, e = np.frexp(max_val)
    exp = bits_per_element - 2 - e

    max_accu_bit_growth = VPU_accu_bits - (2*bits_per_element - 1 - VPU_shr)
    if max_float_accu > 2**max_accu_bit_growth:
        print("Eeeeek")
        exp_modify = int(np.ceil(np.log2(max_float_accu / max_accu_bit_growth)))
        # print('exp_modify', exp_modify)
        # scale the coefs down 
        # and increase the shift left
        # exp -= (exp_modify)
        # TODO
        pass

    fh.write('int32_t __attribute__((aligned (8))) ' + coef_data_name + '[' + str(len(coefs_padded)) + '] = {\n')
    counter = 1
    quantised_coefs = []
    for val in coefs_padded:
        int_val = np.int32(np.rint(np.ldexp(val, exp)))
        int_val = np.clip(int_val, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
        quantised_coefs.append(int_val)
        fh.write('%12d'%(int_val))
        if counter != len(coefs_padded):
            fh.write(',\t')
        if counter % 4 == 0:
            fh.write('\n')
        counter += 1
    fh.write('};\n')

    quantised_coefs = np.array(quantised_coefs, dtype=np.int64)

    v = np.where(quantised_coefs > 0, np.iinfo(np.int32).max, np.iinfo(np.int32).min)
    v = np.array(v, dtype=np.int64)

    accu = 0
    for x, y in zip(v, quantised_coefs):
        accu += np.int64(np.rint((x*y)/2**VPU_shr))
    print(accu)

    if VPU_shr-exp > 0:
        accu_shr = 0
        accu_shl = exp - VPU_shr
    else:
        accu_shr = exp - VPU_shr
        accu_shl = 0

    print('accu_shr', accu_shr, 'accu_shl', accu_shl)

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


def process_array(coefs, filter_name, output_path, gain_dB = 0.0, debug = False, block_length = 8):

    output_file_name = os.path.join(output_path, filter_name + '.h')

    original_filter_length = len(coefs)

    # this is the above but rounded up to the nearest block_length
    target_filter_bank_length = ((original_filter_length + block_length - 1)//block_length)*block_length

    if original_filter_length != target_filter_bank_length:
        print("Warning: ", filter_name, " will be zero padded to length ", target_filter_bank_length)
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
            emit_debug_filter(fh, coefs, filter_name)

            fh.write("#define debug_" + filter_name + "_DATA_BUFFER_ELEMENTS (" + str(len(coefs)) + ")\n")
            fh.write("\n")

        # emit the data define
        data_block_count = filter_block_count + 2
        fh.write("//This is the count of int32_t words to allocate for one data channel.\n")
        fh.write("//i.e. int32_t channel_data[" + filter_name + "_DATA_BUFFER_ELEMENTS] = \{0\};\n")
        fh.write("#define " + filter_name + "_DATA_BUFFER_ELEMENTS (" + str(data_block_count*block_length) + ")\n\n")
    pass

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