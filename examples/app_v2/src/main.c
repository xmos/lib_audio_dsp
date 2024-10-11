#include "dsp_pipeline.h"


#include <xcore/parallel.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>

// from generated code
adsp_generated_t* adsp_generated_init(void);
void adsp_generated_main(void);

static adsp_generated_t* m_dsp;

static void dsp_main(void* p) {adsp_generated_main();}

static void source_sink(void* p) {
  int32_t val=0, output;
  while(true) {
    adsp_source(m_dsp, 0, &val);
    adsp_sink(m_dsp, 0, &output);
    printf("output\t\t\t\t%ld\n", output);
    val++;
  }
}

int main() {
  m_dsp = adsp_generated_init();

  PAR_FUNCS(
    PFUNC(dsp_main, NULL),
    PFUNC(source_sink, NULL)
  );
}
