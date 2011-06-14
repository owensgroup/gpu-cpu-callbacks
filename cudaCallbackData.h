#ifndef __CUDACALLBACKDATA_H__
#define __CUDACALLBACKDATA_H__

#include <cudaCallbackConstants.h>
#include <cudaValue.h>
#include <cudaHostFunction.h>

// some internal data for the callback library. don't touch this.

typedef struct cudaCallbackData
{
  cudaValue_t         cudaParamValues         [CUDA_MAX_CONCURRENT_FUNCS][CUDA_MAX_FUNCTION_PARAMS];
  cudaHostFunction_t  cudaParamFuncs          [CUDA_MAX_CONCURRENT_FUNCS];
  int                 cudaFuncReadyFlag       [CUDA_MAX_CONCURRENT_FUNCS];
  int                 cudaParamFreeIndices    [CUDA_MAX_CONCURRENT_FUNCS];
  int                 cudaParamFreeIndex;
} cudaCallbackData_t;

#endif
