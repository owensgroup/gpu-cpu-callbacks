#ifndef __CUDAPARAMERRORS_H__
#define __CUDAPARAMERRORS_H__

#include <driver_types.h>

extern "C"
{
  // Some more errors to add to the CUDA runtime.
  static const cudaError_t cudaErrorInvalidType             = static_cast<cudaError_t>(0x10000);
  static const cudaError_t cudaErrorArrayUnspecified        = static_cast<cudaError_t>(0x10001);
  static const cudaError_t cudaErrorArrayParamsConflict     = static_cast<cudaError_t>(0x10002);
  static const cudaError_t cudaInvalidHostFunctionPointer   = static_cast<cudaError_t>(0x10003);
  static const cudaError_t cudaInvalidFunctionPointer       = static_cast<cudaError_t>(0x10004);
  static const cudaError_t cudaNullParameterList            = static_cast<cudaError_t>(0x10005);
  static const cudaError_t cudaErrorInvalidReturnType       = static_cast<cudaError_t>(0x10006);
  static const cudaError_t cudaInvalidParameterType         = static_cast<cudaError_t>(0x10100);
  static const cudaError_t cudaInvalidParameterLengthType   = static_cast<cudaError_t>(0x10500);

}

#endif
