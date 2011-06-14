#ifndef __CUDASAFECALL_H__
#define __CUDASAFECALL_H__

#define CUDA_SAFE_CALL(x)                                                                                 \
{                                                                                                         \
  printf("%s.%s.%d: %s\n", __FILE__, __FUNCTION__, __LINE__, #x); fflush(stdout);                         \
  cudaError_t error = (x);                                                                                \
  if (error != cudaSuccess && error != cudaErrorNotReady)                                                 \
  {                                                                                                       \
    printf("%s.%s.%d: 0x%x (%s)\n", __FILE__, __FUNCTION__, __LINE__, error, cudaGetErrorString(error));  \
    cudaGetLastError();                                                                                   \
    exit(1);                                                                                              \
  }                                                                                                       \
}                                                                                                         \

#endif
