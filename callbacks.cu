#include <callbacks.h>
#include <Timer.h>
#include <cstdio>

// __device__ callbackData_t * callbackDeviceData;

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN

#include <windows.h>

inline void callbackSleepMS(const int ms)
{
  if (ms < 0) return;
  if (ms == 0) sched_yield();
  if (ms > 0)
  {
    Sleep(ms);
  }
}

#else

#include <unistd.h>

inline void callbackSleepMS(const int ms)
{
  if (ms < 0) return;
  if (ms == 0) sched_yield();
  if (ms > 0)
  {
    usleep(ms * 1000);
  }
}
#endif

static const char * ERROR_NAMES[] =
{
  "cudaSuccess",
  "cudaErrorMissingConfiguration",
  "cudaErrorMemoryAllocation",
  "cudaErrorInitializationError",
  "cudaErrorLaunchFailure",
  "cudaErrorPriorLaunchFailure",
  "cudaErrorLaunchTimeout",
  "cudaErrorLaunchOutOfResources",
  "cudaErrorInvalidDeviceFunction",
  "cudaErrorInvalidConfiguration",
  "cudaErrorInvalidDevice",
  "cudaErrorInvalidValue",
  "cudaErrorInvalidPitchValue",
  "cudaErrorInvalidSymbol",
  "cudaErrorMapBufferObjectFailed",
  "cudaErrorUnmapBufferObjectFailed",
  "cudaErrorInvalidHostPointer",
  "cudaErrorInvalidDevicePointer",
  "cudaErrorInvalidTexture",
  "cudaErrorInvalidTextureBinding",
  "cudaErrorInvalidChannelDescriptor",
  "cudaErrorInvalidMemcpyDirection",
  "cudaErrorAddressOfConstant",
  "cudaErrorTextureFetchFailed",
  "cudaErrorTextureNotBound",
  "cudaErrorSynchronizationError",
  "cudaErrorInvalidFilterSetting",
  "cudaErrorInvalidNormSetting",
  "cudaErrorMixedDeviceExecution",
  "cudaErrorCudartUnloading",
  "cudaErrorUnknown",
  "cudaErrorNotYetImplemented",
  "cudaErrorMemoryValueTooLarge",
  "cudaErrorInvalidResourceHandle",
  "cudaErrorNotReady",
  "cudaErrorInsufficientDriver",
  "cudaErrorSetOnActiveProcess",
  "cudaErrorNoDevice",
  "cudaErrorStartupFailure",
  "cudaErrorApiFailureBase",
};

__host__ void callbackHostInit(callbackHostData_t ** data)
{
  callbackHostData_t * newData = new callbackHostData_t;
  memset(newData, 0, sizeof(*newData));
  newData->capCallbacks = 16;
  newData->callbacks = new callbackDescriptor_t[newData->capCallbacks];
  memset(newData->callbacks, 0, sizeof(callbackDescriptor_t) * newData->capCallbacks);

  CUDA_SAFE_CALL(cudaHostAlloc           (reinterpret_cast<void ** >(&newData->cpuCallbackData), sizeof(callbackData_t), cudaHostAllocMapped | cudaHostAllocPortable));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&newData->gpuCallbackData), newData->cpuCallbackData, 0));
  memset(newData->cpuCallbackData, 0, sizeof(*newData->cpuCallbackData));
  // cudaHostAlloc           (reinterpret_cast<void ** >(&newData->cpuCallbackData), sizeof(callbackData_t), cudaHostAllocMapped | cudaHostAllocPortable);
  // cudaHostGetDevicePointer(reinterpret_cast<void ** >(&newData->gpuCallbackData), newData->cpuCallbackData, 0);

  *data = newData;
}
__host__ callbackHandle_t callbackRegister(callbackHostData_t * const data, callbackFuncPtr_t funcPtr, const int numParams)
{
  callbackDescriptor_t & cbDesc = data->callbacks[data->numCallbacks++];
  cbDesc.id = data->numCallbacks - 1;
  cbDesc.func = funcPtr;
  cbDesc.numParams = numParams;
  return cbDesc.id;
}
__host__ void callbackCheckRunningThreads(callbackHostData_t  * const data, const bool waitEvenIfNotFinished)
{
  for (int i = 0; i < CUDA_MAX_CONCURRENT_FUNCS; ++i)
  {
    if (data->threads[i] != NULL && (data->threadData[i]->finished || waitEvenIfNotFinished))
    {
      printf("%.3f - %s.%s.%d\n", globalTimer.getElapsedSeconds(), __FILE__, __FUNCTION__, __LINE__); fflush(stdout);
      data->cpuCallbackData->callbackInstances[i].callbackReadyFlag = 0;
      pthread_join(*data->threads[i], NULL);
      delete data->threads[i];
      delete data->threadData[i];
      data->threads[i] = NULL;
      data->threadData[i] = NULL;
    }
  }
}
__host__ void callbackCheckForNewExecutionRequests(callbackHostData_t  * const data)
{
  for (int i = 0; i < CUDA_MAX_CONCURRENT_FUNCS; ++i)
  {
    if (data->threads[i] == NULL && data->cpuCallbackData->callbackInstances[i].callbackReadyFlag != 0)
    {
      callbackInstance_t instance = data->cpuCallbackData->callbackInstances[i]; // yes, we actually mean copy, not reference.
      callbackThreadData_t * cbdata = new callbackThreadData_t;
      cbdata->finished = false;
      memcpy(cbdata->vals, instance.callbackParamValues, sizeof(instance.callbackParamValues));
      cbdata->func = data->callbacks[instance.callbackFunc];

      if (instance.callbackInlineFlag)
      {
        if (false)
        {
          printf("%.3f - %s.%s.%d - flags: \n", globalTimer.getElapsedSeconds(), __FILE__, __FUNCTION__, __LINE__); fflush(stdout);
          int temp = 0;
          while (temp < CUDA_MAX_CONCURRENT_FUNCS)
          {
            printf("  %d", data->cpuCallbackData->callbackInstances[temp].callbackReadyFlag);
            ++temp;
            if (temp % 10 == 0)
            {
              printf("\n");
              if (temp < CUDA_MAX_CONCURRENT_FUNCS)
              {
                printf("%.3f - %s.%s.%d - ", globalTimer.getElapsedSeconds(), __FILE__, __FUNCTION__, __LINE__); fflush(stdout);
              }
            }
          }
          printf("\n");
        }
        // printf("%.3f - %s.%s.%d\n", globalTimer.getElapsedSeconds(), __FILE__, __FUNCTION__, __LINE__); fflush(stdout);
        // printf("%.3f - %s.%s.%d\n", globalTimer.getElapsedSeconds(), __FILE__, __FUNCTION__, __LINE__); fflush(stdout);
        callbackThreadFunc(cbdata);
        delete cbdata;
        data->cpuCallbackData->callbackInstances[i].callbackReadyFlag = 0;
      }
      else
      {
        // printf("%.3f - %s.%s.%d\n", globalTimer.getElapsedSeconds(), __FILE__, __FUNCTION__, __LINE__); fflush(stdout);
        data->threads[i] = new pthread_t;
        data->threadData[i] = cbdata;
        pthread_create(data->threads[i], NULL, callbackThreadFunc, cbdata);
      }
    }
  }
}
__host__ int callbackQuery(callbackHostData_t  * const data, const cudaStream_t kernelStream)
{
  callbackCheckRunningThreads(data, false);
  callbackCheckForNewExecutionRequests(data);

  cudaError_t err = cudaStreamQuery(kernelStream);
  // CUDA_SAFE_CALL(err); // this will pour out text too fast.
  switch (err)
  {
  case cudaSuccess:
    printf("%.3f - kernel finished, checking for any dying callback requests.\n", globalTimer.getElapsedSeconds()); fflush(stdout);
    callbackCheckForNewExecutionRequests(data);
    printf("%.3f - waiting for all threads to finish.\n", globalTimer.getElapsedSeconds()); fflush(stdout);
    callbackCheckRunningThreads(data, true);
    return 1;
  case cudaErrorNotReady:
    break;
  default:
    printf("error %d - %s\n", err, ERROR_NAMES[err]);
    fflush(stdout);
    exit(1);
  }
  return 0;
}
__host__ void callbackSynchronize(callbackHostData_t * const data, const cudaStream_t kernelStream, const int sleepMS)
{
  while (!callbackQuery(data, kernelStream))
  {
    callbackSleepMS(sleepMS);
  }
}
__host__ void * callbackThreadFunc(void * param)
{
  callbackThreadData_t * data = reinterpret_cast<callbackThreadData_t * >(param);
  data->func.func(data->vals);
  data->finished = true;

  return NULL;
}
