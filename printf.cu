#include <systemSafeCall.h>
#include <cudaSafeCall.h>
#include <cudaParams.h>
#include <cudaParamCreate.h>
#include <cudaCallFunc.h>
#include <cstdio>
#include <pthread.h>
#include <sched.h>

#ifndef _WIN32
  #define __cdecl
#endif

extern "C"
{
  void __cdecl printfMarshall(void * retPtr, void * params[])
  {
    static int counter = 0;
    float address[] = { 0.0f, 1.0f, 2.0f };
    printf("printf example, %d %8.8x %8.8x %c\n", *reinterpret_cast<int   * >(params[1]),
                                                  *reinterpret_cast<int   * >(params[0]),
                                                  *reinterpret_cast<int   * >(address + counter),
                                                  *reinterpret_cast<char  * >(params[2]));
    fflush(stdout);
    ++counter;
  }
}

__global__ void mainKernel(cudaCallbackData * cdata, cudaHostFunction_t printfFunc)
{
  cudaCallFunc<int>(cdata, printfFunc, 0.0f, 3, 'a');
  cudaCallFunc<int>(cdata, printfFunc, 1.0f, 4, 'b');
  cudaCallFunc<int>(cdata, printfFunc, 2.0f, 5, 'c');
}

int main(int argc, char ** argv)
{
  dim3 gs = dim3(2, 1, 1);
  dim3 bs = dim3(1, 1, 1);

  cudaParamType_t returnParam;
  cudaParamType_t params[3];
  cudaHostFunction_t printfFunc;

  cudaCallbackData_t * callbackData;

  CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_SAFE_CALL(cudaSetDevice(0));
  CUDA_SAFE_CALL(cudaHostCallbackInit(&callbackData));

  CUDA_SAFE_CALL(cudaParamCreate<int>(&returnParam));
  CUDA_SAFE_CALL(cudaParamCreate<float> (params + 0));
  CUDA_SAFE_CALL(cudaParamCreate<int>   (params + 1));
  CUDA_SAFE_CALL(cudaParamCreate<char>  (params + 2));
  CUDA_SAFE_CALL(cudaCreateHostFunc(&printfFunc, (void * )printfMarshall, returnParam, 3, params));

  mainKernel<<<1, 1, 0, 0>>>(callbackData, printfFunc);
  CUDA_SAFE_CALL(cudaCallbackSynchronize(0));
  return 0;
}
