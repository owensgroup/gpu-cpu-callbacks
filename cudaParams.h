#ifndef __CUDAPARAMS_H__
#define __CUDAPARAMS_H__

#include <cudaCallbackData.h>
#include <cudaHostFunction.h>
#include <cudaParam.h>
#include <cudaParamErrors.h>
#include <cudaParamType.h>
#include <driver_types.h>

// Necessary functions for creating callbacks.

extern "C"
{
  /*
    Registers a new callback.

    func - the place to store the callback info.
    funcPtr - the function pointer for the callback.
    returnParam - the return type of the callback.
    numParams - the number of function parameters of the callback.
    params - the types of each function parameter of the callback.
  */
  cudaError_t cudaCreateHostFunc(cudaHostFunction_t * const func, void * const funcPtr, const cudaParamType_t & returnParam, const int numParams, cudaParamType_t * params);
  // Initializes the callback library and stores a necessary pointer within callbackData.
  cudaError_t cudaHostCallbackInit(cudaCallbackData_t ** callbackData);
  // Sync up execution of the host side until the stream is done. Handles callback requests during sync process.
  cudaError_t cudaCallbackSynchronize(const cudaStream_t stream);

}

extern "C++"
{
  // Create a parameter by type.
  cudaError_t cudaParamCreate(cudaParamType_t * const param,
                              const cudaParamType_t type);
}

#endif
