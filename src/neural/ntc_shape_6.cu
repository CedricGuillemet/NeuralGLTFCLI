// Kernels of network shape 6 (see NTC_SHAPES in ntc_cuda.cu); a TU of its own keeps the device compilations parallel.
#ifndef NTC_ONLY_V2
#define NTC_SHAPE_TU 6
#include "ntc_cuda.cu"
#endif
