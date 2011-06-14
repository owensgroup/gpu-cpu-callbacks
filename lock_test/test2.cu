#include <cstdio>
#include <cstdlib>

#define CUDA_SAFE_CALL(x) { cudaError_t error = (x); if (error != cudaSuccess) { fprintf(stderr, "%s.%s.%d: 0x%x (%s)\n", __FILE__, __FUNCTION__, __LINE__, error, cudaGetErrorString(error)); cudaGetLastError(); exit(1); } }

__device__ unsigned int lock;
__device__ volatile unsigned int counter;

__global__ void kernel()
{
  if (threadIdx.x % 32 == 0)
  {
    while (atomicExch(&lock, 1) == 1) { }
    ++counter;
    atomicExch(&lock, 0);
  }
}

int main(int argc, char ** argv)
{
  unsigned int * gpuLock, * gpuCounter;
  int cpuLock, cpuCounter;
  CUDA_SAFE_CALL(cudaSetDevice(0));
  CUDA_SAFE_CALL(cudaGetSymbolAddress(reinterpret_cast<void ** >(&gpuLock), lock));
  CUDA_SAFE_CALL(cudaGetSymbolAddress(reinterpret_cast<void ** >(&gpuCounter), counter));

  dim3 gs(1, 1, 1), bs(64, 1, 1);
  CUDA_SAFE_CALL(cudaMemset(gpuLock,    0, sizeof(unsigned int)));
  CUDA_SAFE_CALL(cudaMemset(gpuCounter, 0, sizeof(unsigned int)));
  kernel<<<gs, bs>>>();
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(&cpuLock,     gpuLock,    sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&cpuCounter,  gpuCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));

  return 0;
}
