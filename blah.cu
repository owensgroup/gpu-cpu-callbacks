#include "cudaSafeCall.h"
#include <cstdio>

static __device__ int arr[2];
static __device__ void * ptr;

__global__ void kernel()
{
  int * t1 = (int * )&ptr;
  int * t2 = arr;
  int size = sizeof(ptr);
  while (size > 0)
  {
    *(t2++) = *(t1++);
    size -= sizeof(int);
  }
}

int main()
{
  void * gpuArr;
  void * gpuPtr;
  int cpuArr[2];
  void * cpuPtr;
  cpuPtr = reinterpret_cast<void * >(0x1122334455667788L);
  CUDA_SAFE_CALL(cudaSetDevice(0));
  CUDA_SAFE_CALL(cudaGetSymbolAddress(&gpuArr, arr));
  CUDA_SAFE_CALL(cudaGetSymbolAddress(&gpuPtr, ptr));
  CUDA_SAFE_CALL(cudaMemcpy(gpuPtr, &cpuPtr, sizeof(ptr), cudaMemcpyHostToDevice));
  kernel<<<1, 1>>>();
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(cpuArr, gpuArr, sizeof(arr), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&cpuPtr, gpuPtr, sizeof(ptr), cudaMemcpyDeviceToHost));
  printf("arr[0] arr[1] ptr { 0x%08x 0x%08x %p }\n", cpuArr[0], cpuArr[1], cpuPtr);

  return 0;
}

