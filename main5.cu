#include <cudaSafeCall.h>
#include <cstdio>

__shared__ void * buf;

__global__ void kernel(int * flag1, volatile int * flag2, int ** p)
{
  *(volatile int * )flag1 = 1;
  while (*(volatile int * )flag2 == 0) { }
  int * ptr = *(int ** )p;
  *(volatile int * )flag1 = 1;
  *ptr = 5;
}

int main(int argc, char ** argv)
{
  char * cpuMem, * gpuMem;
  CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_SAFE_CALL(cudaSetDevice(1));

  CUDA_SAFE_CALL(cudaHostAlloc(reinterpret_cast<void ** >(&cpuMem), sizeof(int) * 10, cudaHostAllocMapped));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&gpuMem), cpuMem,      0));

  int * cpuFlag1    = (int *)(cpuMem);
  int * cpuFlag2    = (int *)(cpuMem + sizeof(int));
  int * cpuP        = (int *)(cpuMem + sizeof(int) * 2);
  int * nextPointer = (int *)(cpuMem + sizeof(int) * 2 + sizeof(int * ));

  kernel<<<1, 1>>>((int * )gpuMem, (volatile int * )gpuMem + 1, (int ** )(gpuMem + sizeof(int) * 2));
  while (*(volatile int * )cpuFlag1 == 0) { }
  *(volatile int * )cpuFlag1 = 0;
  *(volatile int ** )cpuP = (volatile int *)(gpuMem + ((char *)nextPointer - cpuMem));
  *(volatile int * )cpuFlag2 = 1;
  while (*(volatile int * )cpuFlag1 == 0) { }
  printf("*cpuP gpuMem { %p %p }\n", *(volatile int ** )cpuP, gpuMem);
  printf("nextp: %d\n", *nextPointer);

  return 0;
}
