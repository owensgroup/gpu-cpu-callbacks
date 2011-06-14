#ifndef __CALLBACKS_H__
#define __CALLBACKS_H__

#include <pthread.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#define CUDA_SAFE_CALL(x)                                                                                                                       \
{                                                                                                                                               \
  printf("%3f - %s.%s.%d: %s\n", globalTimer.getElapsedSeconds(), __FILE__, __FUNCTION__, __LINE__, #x); fflush(stdout);                        \
  cudaError_t error = (x);                                                                                                                      \
  if (error != cudaSuccess && error != cudaErrorNotReady)                                                                                       \
  {                                                                                                                                             \
    printf("%3f - %s.%s.%d: 0x%x (%s)\n", globalTimer.getElapsedSeconds(), __FILE__, __FUNCTION__, __LINE__, error, cudaGetErrorString(error)); \
    cudaGetLastError();                                                                                                                         \
    exit(1);                                                                                                                                    \
  }                                                                                                                                             \
}                                                                                                                                               \

typedef int   callbackHandle_t;
typedef float callbackValue_t;
typedef void (*callbackFuncPtr_t)(callbackValue_t *);

enum
{
  CUDA_MAX_CONCURRENT_FUNCS = 256,
  CUDA_MAX_FUNCTION_PARAMS  = 8,
};

enum
{
  CALLBACK_EXEC_TYPE_SYNC,
  CALLBACK_EXEC_TYPE_ASYNC,
};

typedef struct callbackDescriptor
{
  callbackHandle_t id;
  callbackFuncPtr_t func;
  int numParams;
} callbackDescriptor_t;

typedef struct callbackInstance
{
  callbackValue_t   callbackParamValues[CUDA_MAX_FUNCTION_PARAMS];
  callbackHandle_t  callbackFunc;
  int               callbackExecTypeFlag;
  volatile int      callbackReadyFlag;
  int               callbackInlineFlag;
} callbackInstance_t;

typedef struct callbackData
{
  callbackInstance_t  callbackInstances[CUDA_MAX_CONCURRENT_FUNCS];
} callbackData_t;

typedef struct callbackThreadData
{
  volatile bool finished;
  callbackValue_t vals[CUDA_MAX_FUNCTION_PARAMS];
  callbackDescriptor_t func;
} callbackThreadData_t;

typedef struct callbackHostData
{
  pthread_t              * threads            [CUDA_MAX_CONCURRENT_FUNCS];
  callbackThreadData     * threadData         [CUDA_MAX_CONCURRENT_FUNCS];
  callbackDescriptor_t   * callbacks;
  callbackData_t         * cpuCallbackData;
  callbackData_t         * gpuCallbackData;
  int numCallbacks;
  int capCallbacks;
} callbackHostData_t;

__host__ void             callbackHostInit                    (callbackHostData_t ** const data);
__host__ callbackHandle_t callbackRegister                    (callbackHostData_t  * const data, callbackFuncPtr_t funcPtr, const int numParams);
__host__ void             callbackCheckRunningThreads         (callbackHostData_t  * const data, const bool waitEvenIfNotFinished);
__host__ void             callbackCheckForNewExecutionRequests(callbackHostData_t  * const data);
__host__ int              callbackQuery                       (callbackHostData_t  * const data, const cudaStream_t kernelStream);
__host__ void             callbackSynchronize                 (callbackHostData_t  * const data, const cudaStream_t kernelStream, const int sleepMS = -1);

// don't call this from a user application.
__host__ void *           callbackThreadFunc  (void * param);

__device__ int  callbackIsSlotBusy  (const int slot);
__device__ int  callbackPoll(const int slot);
__device__ void callbackWait(const int slot);

/*

// These are defined below, but their declarations are included here for easy reference.

__device__ void callbackExecuteSync (const bool hostInline, const int slot, const callbackHandle_t func);
__device__ void callbackExecuteSync (const bool hostInline, const int slot, const callbackHandle_t func, const float f0);
__device__ void callbackExecuteSync (const bool hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1);
__device__ void callbackExecuteSync (const bool hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2);
__device__ void callbackExecuteSync (const bool hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3);
__device__ void callbackExecuteSync (const bool hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3, const float f4);
__device__ void callbackExecuteSync (const bool hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3, const float f4, const float f5);
__device__ void callbackExecuteSync (const bool hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3, const float f4, const float f5, const float f6);
__device__ void callbackExecuteSync (const bool hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3, const float f4, const float f5, const float f6, const float f7);

__device__ void callbackExecuteAsync(const bool hostInline, const int slot, const callbackHandle_t func);
__device__ void callbackExecuteAsync(const bool hostInline, const int slot, const callbackHandle_t func, const float f0);
__device__ void callbackExecuteAsync(const bool hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1);
__device__ void callbackExecuteAsync(const bool hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2);
__device__ void callbackExecuteAsync(const bool hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3);
__device__ void callbackExecuteAsync(const bool hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3, const float f4);
__device__ void callbackExecuteAsync(const bool hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3, const float f4, const float f5);
__device__ void callbackExecuteAsync(const bool hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3, const float f4, const float f5, const float f6);
__device__ void callbackExecuteAsync(const bool hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3, const float f4, const float f5, const float f6, const float f7);

*/

/** Device Function Definitions. **/

__device__ callbackData_t * callbackDeviceData;

__device__ int  callbackIsSlotBusy  (const int slot)
{
  return callbackDeviceData->callbackInstances[slot].callbackReadyFlag == 1;
}
__device__ int  callbackPoll(const int slot)
{
  return !callbackIsSlotBusy(slot);
}
__device__ void callbackWait(const int slot)
{
  while (callbackIsSlotBusy(slot))
  {
    __threadfence();
  }
}
__device__ void callbackExecute(const int hostInline, const int slot, const callbackHandle_t func, const int type)
{
  callbackDeviceData->callbackInstances[slot].callbackFunc          = func;
  callbackDeviceData->callbackInstances[slot].callbackExecTypeFlag  = type;
  callbackDeviceData->callbackInstances[slot].callbackInlineFlag    = hostInline;
  __threadfence();
  callbackDeviceData->callbackInstances[slot].callbackReadyFlag     = 1;
  __threadfence();
  if (type == CALLBACK_EXEC_TYPE_SYNC)
  {
    callbackWait(slot);
  }
}
__device__ void callbackExecuteSync (const int hostInline, const int slot, const callbackHandle_t func)
{
  callbackExecute(hostInline, slot, func, CALLBACK_EXEC_TYPE_SYNC);
}
__device__ void callbackExecuteSync (const int hostInline, const int slot, const callbackHandle_t func, const float f0)
{
  callbackDeviceData->callbackInstances[slot].callbackParamValues[0] = f0;
  callbackExecuteSync(hostInline, slot, func);
}
__device__ void callbackExecuteSync (const int hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1)
{
  callbackDeviceData->callbackInstances[slot].callbackParamValues[1] = f1;
  callbackExecuteSync(hostInline, slot, func, f0);
}
__device__ void callbackExecuteSync (const int hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2)
{
  callbackDeviceData->callbackInstances[slot].callbackParamValues[2] = f2;
  callbackExecuteSync(hostInline, slot, func, f0, f1);
}
__device__ void callbackExecuteSync (const int hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3)
{
  callbackDeviceData->callbackInstances[slot].callbackParamValues[3] = f3;
  callbackExecuteSync(hostInline, slot, func, f0, f1, f2);
}
__device__ void callbackExecuteSync (const int hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3, const float f4)
{
  callbackDeviceData->callbackInstances[slot].callbackParamValues[4] = f4;
  callbackExecuteSync(hostInline, slot, func, f0, f1, f2, f3);
}
__device__ void callbackExecuteSync (const int hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3, const float f4, const float f5)
{
  callbackDeviceData->callbackInstances[slot].callbackParamValues[5] = f5;
  callbackExecuteSync(hostInline, slot, func, f0, f1, f2, f3, f4);
}
__device__ void callbackExecuteSync (const int hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3, const float f4, const float f5, const float f6)
{
  callbackDeviceData->callbackInstances[slot].callbackParamValues[6] = f6;
  callbackExecuteSync(hostInline, slot, func, f0, f1, f2, f3, f4, f5);
}
__device__ void callbackExecuteSync (const int hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3, const float f4, const float f5, const float f6, const float f7)
{
  callbackDeviceData->callbackInstances[slot].callbackParamValues[7] = f7;
  callbackExecuteSync(hostInline, slot, func, f0, f1, f2, f3, f4, f5, f6);
}

__device__ void callbackExecuteAsync (const int hostInline, const int slot, const callbackHandle_t func)
{
  callbackExecute(hostInline, slot, func, CALLBACK_EXEC_TYPE_ASYNC);
}
__device__ void callbackExecuteAsync (const int hostInline, const int slot, const callbackHandle_t func, const float f0)
{
  callbackDeviceData->callbackInstances[slot].callbackParamValues[0] = f0;
  callbackExecuteAsync(hostInline, slot, func);
}
__device__ void callbackExecuteAsync (const int hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1)
{
  callbackDeviceData->callbackInstances[slot].callbackParamValues[1] = f1;
  callbackExecuteAsync(hostInline, slot, func, f0);
}
__device__ void callbackExecuteAsync (const int hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2)
{
  callbackDeviceData->callbackInstances[slot].callbackParamValues[2] = f2;
  callbackExecuteAsync(hostInline, slot, func, f0, f1);
}
__device__ void callbackExecuteAsync (const int hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3)
{
  callbackDeviceData->callbackInstances[slot].callbackParamValues[3] = f3;
  callbackExecuteAsync(hostInline, slot, func, f0, f1, f2);
}
__device__ void callbackExecuteAsync (const int hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3, const float f4)
{
  callbackDeviceData->callbackInstances[slot].callbackParamValues[4] = f4;
  callbackExecuteAsync(hostInline, slot, func, f0, f1, f2, f3);
}
__device__ void callbackExecuteAsync (const int hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3, const float f4, const float f5)
{
  callbackDeviceData->callbackInstances[slot].callbackParamValues[5] = f5;
  callbackExecuteAsync(hostInline, slot, func, f0, f1, f2, f3, f4);
}
__device__ void callbackExecuteAsync (const int hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3, const float f4, const float f5, const float f6)
{
  callbackDeviceData->callbackInstances[slot].callbackParamValues[6] = f6;
  callbackExecuteAsync(hostInline, slot, func, f0, f1, f2, f3, f4, f5);
}
__device__ void callbackExecuteAsync (const int hostInline, const int slot, const callbackHandle_t func, const float f0, const float f1, const float f2, const float f3, const float f4, const float f5, const float f6, const float f7)
{
  callbackDeviceData->callbackInstances[slot].callbackParamValues[7] = f7;
  callbackExecuteAsync(hostInline, slot, func, f0, f1, f2, f3, f4, f5, f6);
}

#endif
