#include "dsp/td_block_fir.h"

void td_block_fir_data_init(td_block_fir_data_t *d,
                            int32_t *data, uint32_t data_buffer_elements)
{
    d->data = data;
    d->index = 32;
    d->data_stride = (data_buffer_elements*sizeof(int32_t)) - 32;
}

