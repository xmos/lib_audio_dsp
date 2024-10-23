import numpy as np

def quant(coefs, exp):
    quantised = np.rint(np.ldexp(coefs, exp))
    quantised_and_clipped = np.clip(quantised, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    assert np.allclose(quantised, quantised_and_clipped)
    return np.array(quantised_and_clipped, dtype=np.int64)

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

    fh.write('#include "dsp/fir.h"\n')
    fh.write("td_reference_fir_filter_t " + struct_name + ' = {\n')
    fh.write('\t.coefs = '+ coef_data_name +',\n')
    fh.write('\t.length = ' + str(filter_length) +',\n')
    fh.write('\t.exponent = ' + str(-exp) + ',\n')
    fh.write('\t.accu_shr = ' + str(accu_shr) + ',\n')
    fh.write('\t.prod_shr = ' + str(prod_shr) + ',\n')
    fh.write("};\n")
    fh.write("\n")

    return struct_name