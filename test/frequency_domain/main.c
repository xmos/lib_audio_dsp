#include "frequency_domain_pipeline.h"

int main()
{
    adsp_pipeline_t * pipeline = init_dsp();

    while (1){
        adsp_auto_pipeline_main(pipeline);
    }
}