#define CUDA_SAFE_CALL(x)                                                                                         \
{                                                                                                                 \
  printf("%s.%s.%d: %s\n", __FILE__, __FUNCTION__, __LINE__, #x); fflush(stdout);                                 \
  cudaError_t error = (x);                                                                                        \
  if (error != cudaSuccess)                                                                                       \
  {                                                                                                               \
    fprintf(stderr, "%s.%s.%d: 0x%x (%s)\n", __FILE__, __FUNCTION__, __LINE__, error, cudaGetErrorString(error)); \
    cudaGetLastError();                                                                                           \
    exit(1);                                                                                                      \
  }                                                                                                               \
}                                                                                                                 \

#include <cstdio>

int main(int argc, char ** argv)
{
  dim3 gs(5, 5, 1);
  dim3 bs(4, 4, 4);
  int * cpuMem, * gpuMem;

  CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_SAFE_CALL(cudaSetDevice(0));
  CUDA_SAFE_CALL(cudaHostAlloc(reinterpret_cast<void ** >(&cpuMem), sizeof(int) * 25 * 64, cudaHostAllocMapped));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&gpuMem), cpuMem, 0));

  fprintf(stderr, "cpuMem gpuMem { %p %p }\n", cpuMem, gpuMem); fflush(stderr);

  return 0;
}
