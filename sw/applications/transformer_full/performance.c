#include "performance.h"

// For the timer
#include "rv_timer.h"
#include "soc_ctrl.h"
#include "core_v_mini_mcu.h"

// Timer
static rv_timer_t          timer;

void kcom_perfRecordStart( kcom_time_diff_t *perf )
{
    timeStart( perf );
}

void kcom_perfRecordStop( kcom_time_diff_t *perf )
{
    timeStop( perf );
}

void timeStart( kcom_time_diff_t *perf )
{
    perf->start_cy = getTime_cy();
}

void timeStop( kcom_time_diff_t *perf )
{
    perf->end_cy = getTime_cy();
    perf->spent_cy += perf->end_cy - perf->start_cy;
}

uint64_t getTime_cy( )
{
    static uint64_t out;
    rv_timer_counter_read( &timer, HART_ID, &out );
    return out;
}

