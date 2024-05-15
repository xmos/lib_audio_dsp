#include "print.h"
#include "adsp_control.h"
#include "adsp_pipeline.h"
#include "adsp_instance_id_auto.h"
#include "xcore/hwtimer.h"

void send_control_cmds(adsp_pipeline_t * m_dsp, chanend_t c_control);