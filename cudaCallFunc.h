#ifndef __CUDACALLFUNC_H__
#define __CUDACALLFUNC_H__

#include <device_functions.h>
#include <cudaCallbackTypes.h>
#include <cudaSafeCall.h>
#include <cudaParams.h>
#include <cudaValue.h>
#include <cstring>
#include <cstdio>
#include <pthread.h>

// A bunch of functions for actually calling into the library.

const cudaParamType_t * cudaCallbackGetParams(const cudaHostFunction_t & func, size_t & numParams);
const cudaParamType_t * cudaCallbackGetReturnType(const cudaHostFunction_t & func);
const void * cudaCallbackGetFuncPtr(const cudaHostFunction_t & func);

extern "C"
{

  __host__ void cudaCallbackCoerceArguments(char * storage, const cudaParamType_t * dstType, const cudaValue_t * srcValue);
  static cudaCallbackData_t * callbackData = NULL;

  static __device__ unsigned int  cudaParamLock = 0;

  #define cudaFuncTestAndSet(x) atomicExch(x, 1)
  #define cudaFuncUnset(x)      __threadfence(); atomicExch(x, 0)

  inline __host__ cudaError_t cudaHostCallbackInit(cudaCallbackData_t ** data)
  {
    CUDA_SAFE_CALL(cudaHostAlloc(reinterpret_cast<void ** >(&callbackData), sizeof(cudaCallbackData), cudaHostAllocMapped));
    CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(data), callbackData, 0));
    memset(callbackData, 0, sizeof(cudaCallbackData));

    for (int i = 0; i < CUDA_MAX_CONCURRENT_FUNCS; ++i) callbackData->cudaParamFreeIndices[i] = i;

    return cudaSuccess;
  }
  inline __host__ cudaError_t cudaCallbackDump()
  {
    return cudaSuccess;
  }

  typedef struct cudaCallbackThreadData
  {
    bool finished;
    cudaValue_t vals[CUDA_MAX_FUNCTION_PARAMS];
    cudaHostFunction_t func;
  } cudaCallbackThreadData_t;

  __host__ void * cudaCallbackThreadFunc(void * param)
  {
    cudaCallbackThreadData_t * ctd = reinterpret_cast<cudaCallbackThreadData_t * >(param);
    size_t numParams;
    cudaHostFunction_t func = ctd->func;
    cudaValue_t * vals = ctd->vals;
    const cudaParamType_t * params = cudaCallbackGetParams(func, numParams);
    void (*funcPtr)(volatile void *, void *[]) = (void (*)(volatile void *, void *[]))cudaCallbackGetFuncPtr(func);
    void * arr[CUDA_MAX_FUNCTION_PARAMS];
    arr[0] = const_cast<char * >(vals[0].mem);
    for (size_t i = 0; i < numParams; ++i)
    {
      arr[i] = const_cast<char * >(vals[i].mem);
      cudaCallbackCoerceArguments(const_cast<char * >(vals[i].mem), params + i, vals + i);
    }
    funcPtr(const_cast<volatile void * >(arr[0]), arr);
    // __builtin_ia32_sfence();
    ctd->finished = true;

    return NULL; // just in case.
  }
  inline __host__ cudaError_t cudaCallbackSynchronize(const cudaStream_t kernelStream)
  {
    const int                   MAX_THREADS = 10;
    int                         numWorkerThreads = 0;
    pthread_t                   threads     [MAX_THREADS];
    pthread_t                 * freeThreads [MAX_THREADS];
    pthread_t                 * usedThreads [CUDA_MAX_CONCURRENT_FUNCS];
    bool                        working     [CUDA_MAX_CONCURRENT_FUNCS] = { false };
    cudaCallbackThreadData      threadData  [CUDA_MAX_CONCURRENT_FUNCS];

    int count = 0;
    for (int i = 0; i < MAX_THREADS; ++i) freeThreads[i] = threads + i;

    bool keep = true;
    while (keep)
    {
      for (int i = 0; i < CUDA_MAX_CONCURRENT_FUNCS; ++i)
      {
        if (!working[i] && numWorkerThreads < MAX_THREADS && callbackData->cudaFuncReadyFlag[i]/* *const_cast<volatile int * >(callbackData->cudaFuncReadyFlag + i) */)
        {
          memcpy(threadData[i].vals, callbackData->cudaParamValues[i], sizeof(threadData[i].vals));
          threadData[i].func = callbackData->cudaParamFuncs[i];
          threadData[i].finished = false;

          working[i] = true;
          pthread_create(freeThreads[numWorkerThreads], NULL, cudaCallbackThreadFunc, threadData + i);
          usedThreads[i] = freeThreads[numWorkerThreads++];
          ++count;
          // printf("count: %d\n", count); fflush(stdout);
        }
      }
      for (int i = 0; i < CUDA_MAX_CONCURRENT_FUNCS; ++i)
      {
        if (working[i] && threadData[i].finished)
        {
          working[i] = false;
          freeThreads[--numWorkerThreads] = usedThreads[i];
          // callbackData->cudaFuncReadyFlag[i] = false;
          memcpy(&callbackData->cudaParamValues[i][0], &threadData[i].vals[0], sizeof(threadData[i].vals[0]));
          // printf("*(int*)callbackData->cudaParamValues[i][0].mem: %d\n", *(int*)callbackData->cudaParamValues[i][0].mem); fflush(stdout);
          *const_cast<volatile int * >(callbackData->cudaFuncReadyFlag + i) = false;
        }
      }
      switch (cudaStreamQuery(kernelStream))
      {
      case cudaSuccess:
        keep = false;
        break;
      case cudaErrorNotReady:
        keep = true;
        break;
      default:
        CUDA_SAFE_CALL(cudaStreamQuery(kernelStream));
        keep = false;
        break;
      }
    }
    // printf("handled %d requests\n", count); fflush(stdout);

    return cudaCallbackDump();
  }
}

extern "C++"
{
  template <typename T> __host__ __device__ inline cudaParamType_t cudaFuncTypeID()
  {
    return CUDA_TYPE_CHAR_PTR;
  }

  template <> inline cudaParamType_t cudaFuncTypeID<dim3>          () { return CUDA_TYPE_DIM3   ; }
  // template <> inline cudaParamType_t cudaFuncTypeID<size_t>        () { return CUDA_TYPE_SIZET  ; }
  template <> inline cudaParamType_t cudaFuncTypeID<char>          () { return CUDA_TYPE_CHAR   ; }
  template <> inline cudaParamType_t cudaFuncTypeID<unsigned char> () { return CUDA_TYPE_UCHAR  ; }
  template <> inline cudaParamType_t cudaFuncTypeID<short>         () { return CUDA_TYPE_SHORT  ; }
  template <> inline cudaParamType_t cudaFuncTypeID<unsigned short>() { return CUDA_TYPE_USHORT ; }
  template <> inline cudaParamType_t cudaFuncTypeID<int>           () { return CUDA_TYPE_INT    ; }
  template <> inline cudaParamType_t cudaFuncTypeID<unsigned int>  () { return CUDA_TYPE_UINT   ; } // also is size_t
  template <> inline cudaParamType_t cudaFuncTypeID<long>          () { return CUDA_TYPE_LONG   ; }
  template <> inline cudaParamType_t cudaFuncTypeID<unsigned long> () { return CUDA_TYPE_ULONG  ; }
  template <> inline cudaParamType_t cudaFuncTypeID<float>         () { return CUDA_TYPE_FLOAT  ; }
  template <> inline cudaParamType_t cudaFuncTypeID<double>        () { return CUDA_TYPE_DOUBLE ; }

  template <> inline cudaParamType_t cudaFuncTypeID<char2>         () { return CUDA_TYPE_CHAR2  ; }
  template <> inline cudaParamType_t cudaFuncTypeID<uchar2>        () { return CUDA_TYPE_UCHAR2 ; }
  template <> inline cudaParamType_t cudaFuncTypeID<short2>        () { return CUDA_TYPE_SHORT2 ; }
  template <> inline cudaParamType_t cudaFuncTypeID<ushort2>       () { return CUDA_TYPE_USHORT2; }
  template <> inline cudaParamType_t cudaFuncTypeID<int2>          () { return CUDA_TYPE_INT2   ; }
  template <> inline cudaParamType_t cudaFuncTypeID<uint2>         () { return CUDA_TYPE_UINT2  ; }
  template <> inline cudaParamType_t cudaFuncTypeID<long2>         () { return CUDA_TYPE_LONG2  ; }
  template <> inline cudaParamType_t cudaFuncTypeID<ulong2>        () { return CUDA_TYPE_ULONG2 ; }
  template <> inline cudaParamType_t cudaFuncTypeID<float2>        () { return CUDA_TYPE_FLOAT2 ; }
  template <> inline cudaParamType_t cudaFuncTypeID<double2>       () { return CUDA_TYPE_DOUBLE2; }

  template <> inline cudaParamType_t cudaFuncTypeID<char3>         () { return CUDA_TYPE_CHAR3  ; }
  template <> inline cudaParamType_t cudaFuncTypeID<uchar3>        () { return CUDA_TYPE_UCHAR3 ; }
  template <> inline cudaParamType_t cudaFuncTypeID<short3>        () { return CUDA_TYPE_SHORT3 ; }
  template <> inline cudaParamType_t cudaFuncTypeID<ushort3>       () { return CUDA_TYPE_USHORT3; }
  template <> inline cudaParamType_t cudaFuncTypeID<int3>          () { return CUDA_TYPE_INT3   ; }
  template <> inline cudaParamType_t cudaFuncTypeID<uint3>         () { return CUDA_TYPE_UINT3  ; }
  template <> inline cudaParamType_t cudaFuncTypeID<long3>         () { return CUDA_TYPE_LONG3  ; }
  template <> inline cudaParamType_t cudaFuncTypeID<ulong3>        () { return CUDA_TYPE_ULONG3 ; }
  template <> inline cudaParamType_t cudaFuncTypeID<float3>        () { return CUDA_TYPE_FLOAT3 ; }
  template <> inline cudaParamType_t cudaFuncTypeID<double3>       () { return CUDA_TYPE_DOUBLE3; }

  template <> inline cudaParamType_t cudaFuncTypeID<char4>         () { return CUDA_TYPE_CHAR4  ; }
  template <> inline cudaParamType_t cudaFuncTypeID<uchar4>        () { return CUDA_TYPE_UCHAR4 ; }
  template <> inline cudaParamType_t cudaFuncTypeID<short4>        () { return CUDA_TYPE_SHORT4 ; }
  template <> inline cudaParamType_t cudaFuncTypeID<ushort4>       () { return CUDA_TYPE_USHORT4; }
  template <> inline cudaParamType_t cudaFuncTypeID<int4>          () { return CUDA_TYPE_INT4   ; }
  template <> inline cudaParamType_t cudaFuncTypeID<uint4>         () { return CUDA_TYPE_UINT4  ; }
  template <> inline cudaParamType_t cudaFuncTypeID<long4>         () { return CUDA_TYPE_LONG4  ; }
  template <> inline cudaParamType_t cudaFuncTypeID<ulong4>        () { return CUDA_TYPE_ULONG4 ; }
  template <> inline cudaParamType_t cudaFuncTypeID<float4>        () { return CUDA_TYPE_FLOAT4 ; }
  template <> inline cudaParamType_t cudaFuncTypeID<double4>       () { return CUDA_TYPE_DOUBLE4; }

  __device__ int cudaFuncAcquireIndex(cudaCallbackData_t * callbackData)
  {
    int ret = -1;

    while (ret == -1)
    {
      while (atomicExch(&cudaParamLock, 1) == 1) { }
      if (callbackData->cudaParamFreeIndex < CUDA_MAX_CONCURRENT_FUNCS)
      {
        ret = callbackData->cudaParamFreeIndices[callbackData->cudaParamFreeIndex++];
      }
      atomicExch(&cudaParamLock, 0);
    }

    return ret;
  }

  __device__ void cudaFuncReleaseIndex(cudaCallbackData_t * callbackData, const int index)
  {
    while (atomicExch(&cudaParamLock, 1) == 1) { }
    callbackData->cudaParamFreeIndices[--callbackData->cudaParamFreeIndex] = index;

    atomicExch(&cudaParamLock, 0);
  }

  template <typename T>
  __device__ inline T cudaFuncSignalCPUAndPoll(cudaCallbackData_t * callbackData, const int index)
  {
    volatile int * flag = const_cast<volatile int * >(callbackData->cudaFuncReadyFlag + index);
    __threadfence();
    *flag = 1;
    __threadfence();
    while (*flag == 1) { }
    T ret = *reinterpret_cast<volatile T * >(callbackData->cudaParamValues[index][0].mem);
    cudaFuncReleaseIndex(callbackData, index);
    return ret;
  }

  __device__ void cudaFuncSignalCPU(cudaCallbackData_t * callbackData, const int index)
  {
    volatile int * flag = const_cast<volatile int * >(callbackData->cudaFuncReadyFlag + index);
    __threadfence();
    *flag = 1;
    __threadfence();
  }

  template <typename T> __device__ bool isfloat()         { return false; }
  template <>           __device__ bool isfloat<float>()  { return true;  }

  #define cudaMarshallParameter(t, val)                                                             \
  {                                                                                                 \
    val.type = cudaFuncTypeID<typeof(t)>();                                                         \
    int size = sizeof(t);                                                                           \
    if (size <= sizeof(int))                                                                        \
    {                                                                                               \
      *reinterpret_cast<volatile int * >(&val.mem[0]) = *reinterpret_cast<const int * >(&t);        \
    }                                                                                               \
    else if (size <= sizeof(void * ))                                                               \
    {                                                                                               \
      *reinterpret_cast<void * volatile * >(&val.mem[0]) = *reinterpret_cast<void * const * >(&t);  \
    }                                                                                               \
    else                                                                                            \
    {                                                                                               \
      const int * src = reinterpret_cast<const int * >(&t);                                         \
      volatile int * dst = reinterpret_cast<volatile int * >(val.mem);                              \
      while (size > 0)                                                                              \
      {                                                                                             \
        *(dst++) = *(src++);                                                                        \
        size -= sizeof(int);                                                                        \
        --size;                                                                                     \
      }                                                                                             \
      if (size > 0)                                                                                 \
      {                                                                                             \
        *dst = *src;                                                                                \
      }                                                                                             \
    }                                                                                               \
  }                                                                                                 \

/*
  template <typename T>
  __device__ inline void cudaMarshallParameter(const T & t, cudaValue_t & val)
  {
    val.type = cudaFuncTypeID<T>();

    int size = sizeof(t);
    if (size <= sizeof(int))
    {
      *reinterpret_cast<volatile int * >(&val.mem[0]) = *reinterpret_cast<const int * >(&t);
    }
    else if (size <= sizeof(void * ))
    {
      *reinterpret_cast<void * volatile * >(&val.mem[0]) = *reinterpret_cast<void * const * >(&t);
    }
    else
    {
      const int * src = reinterpret_cast<const int * >(&t);
      volatile int * dst = reinterpret_cast<volatile int * >(val.mem);
      while (size > 0)
      {
        *(dst++) = *(src++);
        size -= sizeof(int);
        --size;
      }
      if (size > 0)
      {
        *dst = *src;
      }
    }
  }
*/

  template <typename R>
  __device__ R cudaCallFunc(cudaCallbackData_t * callbackData, const cudaHostFunction_t & func)
  {
    int index = cudaFuncAcquireIndex(callbackData);
    callbackData->cudaParamFuncs[index] = func;
    return cudaFuncSignalCPUAndPoll<R>(callbackData, index);
  }

  template <typename R, typename T1>
  __device__ R cudaCallFunc(cudaCallbackData_t * callbackData, const cudaHostFunction_t & func, const T1 & t1)
  {
    int index = cudaFuncAcquireIndex(callbackData);
    callbackData->cudaParamFuncs[index] = func;
    cudaMarshallParameter(t1, callbackData->cudaParamValues[index][0]);
    return cudaFuncSignalCPUAndPoll<R>(callbackData, index);
  }

  template <typename R, typename T1, typename T2>
  __device__ R cudaCallFunc(cudaCallbackData_t * callbackData, const cudaHostFunction_t & func, const T1 & t1, const T2 & t2)
  {
    int index = cudaFuncAcquireIndex(callbackData);
    callbackData->cudaParamFuncs[index] = func;
    cudaMarshallParameter(t1, callbackData->cudaParamValues[index][0]);
    cudaMarshallParameter(t2, callbackData->cudaParamValues[index][1]);
    return cudaFuncSignalCPUAndPoll<R>(callbackData, index);
  }

  template <typename R, typename T1, typename T2, typename T3>
  __device__ R cudaCallFunc(cudaCallbackData_t * callbackData, const cudaHostFunction_t & func, const T1 & t1, const T2 & t2, const T3 & t3)
  {
    int index = cudaFuncAcquireIndex(callbackData);
    callbackData->cudaParamFuncs[index] = func;
    cudaMarshallParameter(t1, callbackData->cudaParamValues[index][0]);
    cudaMarshallParameter(t2, callbackData->cudaParamValues[index][1]);
    cudaMarshallParameter(t3, callbackData->cudaParamValues[index][2]);
    return cudaFuncSignalCPUAndPoll<R>(callbackData, index);
  }

  template <typename R, typename T1, typename T2, typename T3, typename T4>
  __device__ R cudaCallFunc(cudaCallbackData_t * callbackData, const cudaHostFunction_t & func, const T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4)
  {
    int index = cudaFuncAcquireIndex(callbackData);
    callbackData->cudaParamFuncs[index] = func;
    cudaMarshallParameter(t1, callbackData->cudaParamValues[index][0]);
    cudaMarshallParameter(t2, callbackData->cudaParamValues[index][1]);
    cudaMarshallParameter(t3, callbackData->cudaParamValues[index][2]);
    cudaMarshallParameter(t4, callbackData->cudaParamValues[index][3]);
    return cudaFuncSignalCPUAndPoll<R>(callbackData, index);
  }

  template <typename R, typename T1, typename T2, typename T3, typename T4, typename T5>
  __device__ R cudaCallFunc(cudaCallbackData_t * callbackData, const cudaHostFunction_t & func, const T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4, const T5 & t5)
  {
    int index = cudaFuncAcquireIndex(callbackData);
    callbackData->cudaParamFuncs[index] = func;
    cudaMarshallParameter(t1, callbackData->cudaParamValues[index][0]);
    cudaMarshallParameter(t2, callbackData->cudaParamValues[index][1]);
    cudaMarshallParameter(t3, callbackData->cudaParamValues[index][2]);
    cudaMarshallParameter(t4, callbackData->cudaParamValues[index][3]);
    cudaMarshallParameter(t5, callbackData->cudaParamValues[index][4]);
    return cudaFuncSignalCPUAndPoll<R>(callbackData, index);
  }

  template <typename R, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
  __device__ R cudaCallFunc(cudaCallbackData_t * callbackData, const cudaHostFunction_t & func, const T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4, const T5 & t5, const T6 & t6)
  {
    int index = cudaFuncAcquireIndex(callbackData);
    callbackData->cudaParamFuncs[index] = func;
    cudaMarshallParameter(t1, callbackData->cudaParamValues[index][0]);
    cudaMarshallParameter(t2, callbackData->cudaParamValues[index][1]);
    cudaMarshallParameter(t3, callbackData->cudaParamValues[index][2]);
    cudaMarshallParameter(t4, callbackData->cudaParamValues[index][3]);
    cudaMarshallParameter(t5, callbackData->cudaParamValues[index][4]);
    cudaMarshallParameter(t6, callbackData->cudaParamValues[index][5]);
    return cudaFuncSignalCPUAndPoll<R>(callbackData, index);
  }

  __device__ callbackAsyncRequest_t cudaCallFuncAsync(cudaCallbackData_t * callbackData, const cudaHostFunction_t & func)
  {
    int index = cudaFuncAcquireIndex(callbackData);
    callbackData->cudaParamFuncs[index] = func;
    cudaFuncSignalCPU(callbackData, index);
    return index;
  }

  template <typename T1>
  __device__ callbackAsyncRequest_t cudaCallFuncAsync(cudaCallbackData_t * callbackData, const cudaHostFunction_t & func, const T1 & t1)
  {
    int index = cudaFuncAcquireIndex(callbackData);
    callbackData->cudaParamFuncs[index] = func;
    cudaMarshallParameter(t1, callbackData->cudaParamValues[index][0]);
    cudaFuncSignalCPU(callbackData, index);
    return index;
  }

  template <typename T1, typename T2>
  __device__ callbackAsyncRequest_t cudaCallFuncAsync(cudaCallbackData_t * callbackData, const cudaHostFunction_t & func, const T1 & t1, const T2 & t2)
  {
    int index = cudaFuncAcquireIndex(callbackData);
    callbackData->cudaParamFuncs[index] = func;
    cudaMarshallParameter(t1, callbackData->cudaParamValues[index][0]);
    cudaMarshallParameter(t2, callbackData->cudaParamValues[index][1]);
    cudaFuncSignalCPU(callbackData, index);
    return index;
  }

  template <typename T1, typename T2, typename T3>
  __device__ callbackAsyncRequest_t cudaCallFuncAsync(cudaCallbackData_t * callbackData, const cudaHostFunction_t & func, const T1 & t1, const T2 & t2, const T3 & t3)
  {
    int index = cudaFuncAcquireIndex(callbackData);
    callbackData->cudaParamFuncs[index] = func;
    cudaMarshallParameter(t1, callbackData->cudaParamValues[index][0]);
    cudaMarshallParameter(t2, callbackData->cudaParamValues[index][1]);
    cudaMarshallParameter(t3, callbackData->cudaParamValues[index][2]);
    cudaFuncSignalCPU(callbackData, index);
    return index;
  }

  template <typename T1, typename T2, typename T3, typename T4>
  __device__ callbackAsyncRequest_t cudaCallFuncAsync(cudaCallbackData_t * callbackData, const cudaHostFunction_t & func, const T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4)
  {
    int index = cudaFuncAcquireIndex(callbackData);
    callbackData->cudaParamFuncs[index] = func;
    cudaMarshallParameter(t1, callbackData->cudaParamValues[index][0]);
    cudaMarshallParameter(t2, callbackData->cudaParamValues[index][1]);
    cudaMarshallParameter(t3, callbackData->cudaParamValues[index][2]);
    cudaMarshallParameter(t4, callbackData->cudaParamValues[index][3]);
    cudaFuncSignalCPU(callbackData, index);
    return index;
  }

  template <typename T1, typename T2, typename T3, typename T4, typename T5>
  __device__ callbackAsyncRequest_t cudaCallFuncAsync(cudaCallbackData_t * callbackData, const cudaHostFunction_t & func, const T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4, const T5 & t5)
  {
    int index = cudaFuncAcquireIndex(callbackData);
    callbackData->cudaParamFuncs[index] = func;
    cudaMarshallParameter(t1, callbackData->cudaParamValues[index][0]);
    cudaMarshallParameter(t2, callbackData->cudaParamValues[index][1]);
    cudaMarshallParameter(t3, callbackData->cudaParamValues[index][2]);
    cudaMarshallParameter(t4, callbackData->cudaParamValues[index][3]);
    cudaMarshallParameter(t5, callbackData->cudaParamValues[index][4]);
    cudaFuncSignalCPU(callbackData, index);
    return index;
  }

  template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
  __device__ callbackAsyncRequest_t cudaCallFuncAsync(cudaCallbackData_t * callbackData, const cudaHostFunction_t & func, const T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4, const T5 & t5, const T6 & t6)
  {
    int index = cudaFuncAcquireIndex(callbackData);
    callbackData->cudaParamFuncs[index] = func;
    cudaMarshallParameter(t1, callbackData->cudaParamValues[index][0]);
    cudaMarshallParameter(t2, callbackData->cudaParamValues[index][1]);
    cudaMarshallParameter(t3, callbackData->cudaParamValues[index][2]);
    cudaMarshallParameter(t4, callbackData->cudaParamValues[index][3]);
    cudaMarshallParameter(t5, callbackData->cudaParamValues[index][4]);
    cudaMarshallParameter(t6, callbackData->cudaParamValues[index][5]);
    cudaFuncSignalCPU(callbackData, index);
    return index;
  }


  template <typename T>
  __device__ inline bool cudaFuncPoll(cudaCallbackData_t * callbackData, const callbackAsyncRequest_t req, T * const returnVal)
  {
    volatile int * flag = const_cast<volatile int * >(callbackData->cudaFuncReadyFlag + req);
    if (*flag == 0)
    {
      if (returnVal) *returnVal = *reinterpret_cast<volatile T * >(callbackData->cudaParamValues[req][0].mem);
      cudaFuncReleaseIndex(callbackData, req);
      return true;
    }
    return false;
  }

  char * cudaParamToString(cudaValue_t & val)
  {
    static int bufIndex = 0;
    static char buf[10][1024];

    volatile char * mem = const_cast<volatile char * >(val.mem);

    switch (val.type)
    {
    case CUDA_TYPE_VOID:    sprintf(buf[bufIndex], "void"); break;
    case CUDA_TYPE_SIZET:   sprintf(buf[bufIndex], "sizet=%lu",               *reinterpret_cast<volatile size_t         * >(mem)); break;
    case CUDA_TYPE_CHAR:    sprintf(buf[bufIndex], "char=%c",                 *reinterpret_cast<volatile char           * >(mem)); break;
    case CUDA_TYPE_UCHAR:   sprintf(buf[bufIndex], "uchar=%c",                *reinterpret_cast<volatile unsigned char  * >(mem)); break;
    case CUDA_TYPE_SHORT:   sprintf(buf[bufIndex], "short=%d",                *reinterpret_cast<volatile short          * >(mem)); break;
    case CUDA_TYPE_USHORT:  sprintf(buf[bufIndex], "ushort=%u",               *reinterpret_cast<volatile unsigned short * >(mem)); break;
    case CUDA_TYPE_INT:     sprintf(buf[bufIndex], "int=%d",                  *reinterpret_cast<volatile int            * >(mem)); break;
    case CUDA_TYPE_UINT:    sprintf(buf[bufIndex], "uint=%u",                 *reinterpret_cast<volatile unsigned int   * >(mem)); break;
    case CUDA_TYPE_LONG:    sprintf(buf[bufIndex], "long=%ld",                *reinterpret_cast<volatile long           * >(mem)); break;
    case CUDA_TYPE_ULONG:   sprintf(buf[bufIndex], "ulong=%lu",               *reinterpret_cast<volatile unsigned long  * >(mem)); break;
    case CUDA_TYPE_FLOAT:   sprintf(buf[bufIndex], "float=%f",                *reinterpret_cast<volatile float          * >(mem)); break;
    case CUDA_TYPE_DOUBLE:  sprintf(buf[bufIndex], "double=%f",               *reinterpret_cast<volatile double         * >(mem)); break;
    case CUDA_TYPE_CHAR2:   sprintf(buf[bufIndex], "char2=%c,%c",             *reinterpret_cast<volatile char           * >(mem), *reinterpret_cast<volatile char           * >(mem + 1)); break;
    case CUDA_TYPE_UCHAR2:  sprintf(buf[bufIndex], "uchar2=%c,%c",            *reinterpret_cast<volatile unsigned char  * >(mem), *reinterpret_cast<volatile unsigned char  * >(mem + 1)); break;
    case CUDA_TYPE_SHORT2:  sprintf(buf[bufIndex], "short2=%d,%d",            *reinterpret_cast<volatile short          * >(mem), *reinterpret_cast<volatile short          * >(mem + 2)); break;
    case CUDA_TYPE_USHORT2: sprintf(buf[bufIndex], "ushort2=%u,%u",           *reinterpret_cast<volatile unsigned short * >(mem), *reinterpret_cast<volatile unsigned short * >(mem + 2)); break;
    case CUDA_TYPE_INT2:    sprintf(buf[bufIndex], "int2=%d,%d",              *reinterpret_cast<volatile int            * >(mem), *reinterpret_cast<volatile int            * >(mem + 4)); break;
    case CUDA_TYPE_UINT2:   sprintf(buf[bufIndex], "uint2=%u,%u",             *reinterpret_cast<volatile unsigned int   * >(mem), *reinterpret_cast<volatile unsigned int   * >(mem + 4)); break;
    case CUDA_TYPE_LONG2:   sprintf(buf[bufIndex], "long2=%ld,%ld",           *reinterpret_cast<volatile long           * >(mem), *reinterpret_cast<volatile long           * >(mem + 4)); break;
    case CUDA_TYPE_ULONG2:  sprintf(buf[bufIndex], "ulong2=%lu,%lu",          *reinterpret_cast<volatile unsigned long  * >(mem), *reinterpret_cast<volatile unsigned long  * >(mem + 4)); break;
    case CUDA_TYPE_FLOAT2:  sprintf(buf[bufIndex], "float2=%f,%f",            *reinterpret_cast<volatile float          * >(mem), *reinterpret_cast<volatile float          * >(mem + 4)); break;
    case CUDA_TYPE_DOUBLE2: sprintf(buf[bufIndex], "double2=%f,%f",           *reinterpret_cast<volatile double         * >(mem), *reinterpret_cast<volatile double         * >(mem + 8)); break;
    case CUDA_TYPE_DIM3:    sprintf(buf[bufIndex], "dim3=%u,%u,%u",           *reinterpret_cast<volatile unsigned int   * >(mem), *reinterpret_cast<volatile unsigned int   * >(mem + 4), *reinterpret_cast<volatile unsigned int   * >(mem +  8)); break;
    case CUDA_TYPE_CHAR3:   sprintf(buf[bufIndex], "char3=%c,%c,%c",          *reinterpret_cast<volatile char           * >(mem), *reinterpret_cast<volatile char           * >(mem + 1), *reinterpret_cast<volatile char           * >(mem +  2)); break;
    case CUDA_TYPE_UCHAR3:  sprintf(buf[bufIndex], "uchar3=%c,%c,%c",         *reinterpret_cast<volatile unsigned char  * >(mem), *reinterpret_cast<volatile unsigned char  * >(mem + 1), *reinterpret_cast<volatile unsigned char  * >(mem +  2)); break;
    case CUDA_TYPE_SHORT3:  sprintf(buf[bufIndex], "short3=%d,%d,%d",         *reinterpret_cast<volatile short          * >(mem), *reinterpret_cast<volatile short          * >(mem + 2), *reinterpret_cast<volatile short          * >(mem +  4)); break;
    case CUDA_TYPE_USHORT3: sprintf(buf[bufIndex], "ushort3=%u,%u,%u",        *reinterpret_cast<volatile unsigned short * >(mem), *reinterpret_cast<volatile unsigned short * >(mem + 2), *reinterpret_cast<volatile unsigned short * >(mem +  4)); break;
    case CUDA_TYPE_INT3:    sprintf(buf[bufIndex], "int3=%d,%d,%d",           *reinterpret_cast<volatile int            * >(mem), *reinterpret_cast<volatile int            * >(mem + 4), *reinterpret_cast<volatile int            * >(mem +  8)); break;
    case CUDA_TYPE_UINT3:   sprintf(buf[bufIndex], "uint3=%u,%u,%u",          *reinterpret_cast<volatile unsigned int   * >(mem), *reinterpret_cast<volatile unsigned int   * >(mem + 4), *reinterpret_cast<volatile unsigned int   * >(mem +  8)); break;
    case CUDA_TYPE_LONG3:   sprintf(buf[bufIndex], "long3=%ld,%ld,%ld",       *reinterpret_cast<volatile long           * >(mem), *reinterpret_cast<volatile long           * >(mem + 4), *reinterpret_cast<volatile long           * >(mem +  8)); break;
    case CUDA_TYPE_ULONG3:  sprintf(buf[bufIndex], "ulong3=%lu,%lu,%lu",      *reinterpret_cast<volatile unsigned long  * >(mem), *reinterpret_cast<volatile unsigned long  * >(mem + 4), *reinterpret_cast<volatile unsigned long  * >(mem +  8)); break;
    case CUDA_TYPE_FLOAT3:  sprintf(buf[bufIndex], "float3=%f,%f,%f",         *reinterpret_cast<volatile float          * >(mem), *reinterpret_cast<volatile float          * >(mem + 4), *reinterpret_cast<volatile float          * >(mem +  8)); break;
    case CUDA_TYPE_DOUBLE3: sprintf(buf[bufIndex], "double3=%f,%f,%f",        *reinterpret_cast<volatile double         * >(mem), *reinterpret_cast<volatile double         * >(mem + 8), *reinterpret_cast<volatile double         * >(mem + 16)); break;
    case CUDA_TYPE_CHAR4:   sprintf(buf[bufIndex], "char4=%c,%c,%c,%c",       *reinterpret_cast<volatile char           * >(mem), *reinterpret_cast<volatile char           * >(mem + 1), *reinterpret_cast<volatile char           * >(mem +  2), *reinterpret_cast<volatile char           * >(mem +  3)); break;
    case CUDA_TYPE_UCHAR4:  sprintf(buf[bufIndex], "uchar4=%c,%c,%c,%c",      *reinterpret_cast<volatile unsigned char  * >(mem), *reinterpret_cast<volatile unsigned char  * >(mem + 1), *reinterpret_cast<volatile unsigned char  * >(mem +  2), *reinterpret_cast<volatile unsigned char  * >(mem +  3)); break;
    case CUDA_TYPE_SHORT4:  sprintf(buf[bufIndex], "short4=%d,%d,%d,%d",      *reinterpret_cast<volatile short          * >(mem), *reinterpret_cast<volatile short          * >(mem + 2), *reinterpret_cast<volatile short          * >(mem +  4), *reinterpret_cast<volatile short          * >(mem +  6)); break;
    case CUDA_TYPE_USHORT4: sprintf(buf[bufIndex], "ushort4=%u,%u,%u,%u",     *reinterpret_cast<volatile unsigned short * >(mem), *reinterpret_cast<volatile unsigned short * >(mem + 2), *reinterpret_cast<volatile unsigned short * >(mem +  4), *reinterpret_cast<volatile unsigned short * >(mem +  6)); break;
    case CUDA_TYPE_INT4:    sprintf(buf[bufIndex], "int4=%d,%d,%d,%d",        *reinterpret_cast<volatile int            * >(mem), *reinterpret_cast<volatile int            * >(mem + 4), *reinterpret_cast<volatile int            * >(mem +  8), *reinterpret_cast<volatile int            * >(mem + 12)); break;
    case CUDA_TYPE_UINT4:   sprintf(buf[bufIndex], "uint4=%u,%u,%u,%u",       *reinterpret_cast<volatile unsigned int   * >(mem), *reinterpret_cast<volatile unsigned int   * >(mem + 4), *reinterpret_cast<volatile unsigned int   * >(mem +  8), *reinterpret_cast<volatile unsigned int   * >(mem + 12)); break;
    case CUDA_TYPE_LONG4:   sprintf(buf[bufIndex], "long4=%ld,%ld,%ld,%ld",   *reinterpret_cast<volatile long           * >(mem), *reinterpret_cast<volatile long           * >(mem + 4), *reinterpret_cast<volatile long           * >(mem +  8), *reinterpret_cast<volatile long           * >(mem + 12)); break;
    case CUDA_TYPE_ULONG4:  sprintf(buf[bufIndex], "ulong4=%lu,%lu,%lu,%lu",  *reinterpret_cast<volatile unsigned long  * >(mem), *reinterpret_cast<volatile unsigned long  * >(mem + 4), *reinterpret_cast<volatile unsigned long  * >(mem +  8), *reinterpret_cast<volatile unsigned long  * >(mem + 12)); break;
    case CUDA_TYPE_FLOAT4:  sprintf(buf[bufIndex], "float4=%f,%f,%f,%f",      *reinterpret_cast<volatile float          * >(mem), *reinterpret_cast<volatile float          * >(mem + 4), *reinterpret_cast<volatile float          * >(mem +  8), *reinterpret_cast<volatile float          * >(mem + 12)); break;
    case CUDA_TYPE_DOUBLE4: sprintf(buf[bufIndex], "double4=%f,%f,%f,%f",     *reinterpret_cast<volatile double         * >(mem), *reinterpret_cast<volatile double         * >(mem + 8), *reinterpret_cast<volatile double         * >(mem + 16), *reinterpret_cast<volatile double         * >(mem + 24)); break;
    default:  sprintf(buf[bufIndex], "%p", *reinterpret_cast<void * volatile * >(mem)); break;
    }
    char * ret = buf[bufIndex];
    bufIndex = (bufIndex + 1) % 10;

    return ret;
  }
}

#endif
