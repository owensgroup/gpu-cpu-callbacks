#define CUDA_SAFE_CALL(x)                                                                                         \
{                                                                                                                 \
  cudaError_t error = (x);                                                                                        \
  if (error != cudaSuccess)                                                                                       \
  {                                                                                                               \
    fprintf(stderr, "%s.%s.%d: 0x%x (%s)\n", __FILE__, __FUNCTION__, __LINE__, error, cudaGetErrorString(error)); \
    cudaGetLastError();                                                                                           \
    exit(1);                                                                                                      \
  }                                                                                                               \
}                                                                                                                 \

#include <cstdio>
#include <windows.h>

void setAffinity()
{
  if (SetProcessAffinityMask(GetCurrentProcess(), 1) == 0)
  {
    DWORD error = GetLastError();
    printf("spam failed: %d\n", static_cast<int>(error));
    fflush(stdout);
    exit(1);
  }
}

inline static LARGE_INTEGER & licast(void * const p) { return *reinterpret_cast<LARGE_INTEGER * >(p); }

class Timer
{
  protected:
    void * begin, * end, * misc;
    char data[128];
  public:
    Timer()
    {
      begin = data;
      end = data + sizeof(LARGE_INTEGER);
      misc = data + sizeof(LARGE_INTEGER) * 2;
      QueryPerformanceFrequency(&licast(misc));
    }
    void start()
    {
      QueryPerformanceCounter(&licast(begin));
    }
    void stop()
    {
      QueryPerformanceCounter(&licast(end));
    }
    float getElapsedSeconds() const
    {
      double diff = static_cast<double>(licast(end).QuadPart - licast(begin).QuadPart);
      return diff / static_cast<float>(licast(misc).QuadPart);
    }
    float getElapsedMilliseconds() const
    {
      return getElapsedSeconds() * 1000.0f;
    }
    float getElapsedMicroseconds() const
    {
      return getElapsedSeconds() * 1000000.0f;
    }
};


__device__ unsigned int lock;
__device__ volatile unsigned int counter;

__global__ void kernel1()
{
  atomicInc(const_cast<unsigned int * >(&counter), 0xFFFFFFFF);
}
__global__ void kernel2()
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
  const int BLOCK_SIZES[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
  const int BLOCK_SIZES_2[] = { 32, 64, 128, 256, 512 };
  const int GRID_SIZES[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
  int NUM_BLOCK_SIZES = sizeof(BLOCK_SIZES) / sizeof(BLOCK_SIZES[0]);
  int NUM_BLOCK_SIZES_2 = sizeof(BLOCK_SIZES_2) / sizeof(BLOCK_SIZES_2[0]);
  int NUM_GRID_SIZES  = sizeof(GRID_SIZES)  / sizeof(GRID_SIZES[0]);
  unsigned int * gpuLock, * gpuCounter;
  int cpuLock, cpuCounter;
  CUDA_SAFE_CALL(cudaSetDevice(0));
  CUDA_SAFE_CALL(cudaGetSymbolAddress(reinterpret_cast<void ** >(&gpuLock), lock));
  CUDA_SAFE_CALL(cudaGetSymbolAddress(reinterpret_cast<void ** >(&gpuCounter), counter));

  printf("  gs/bs");
  for (int i = 0; i < NUM_BLOCK_SIZES; ++i)
  {
    printf("  | %13d", BLOCK_SIZES[i]);
  }
  printf("\n");
  printf("--------");
  for (int i = 0; i < NUM_BLOCK_SIZES; ++i) printf("-+---------------");
  printf("\n");
  for (int i = 0; i < NUM_GRID_SIZES; ++i)
  {
    printf("%7d  | ", GRID_SIZES[i]);
    dim3 gs(GRID_SIZES[i], 1, 1);
    for (int j = 0; j < NUM_BLOCK_SIZES; ++j)
    {
      CUDA_SAFE_CALL(cudaMemset(gpuLock,    0, sizeof(unsigned int)));
      CUDA_SAFE_CALL(cudaMemset(gpuCounter, 0, sizeof(unsigned int)));
      dim3 bs(BLOCK_SIZES[j], 1, 1);
      Timer timer;
      timer.start();
      kernel1<<<gs, bs>>>();
      // kernel2<<<gs, bs>>>();
      CUDA_SAFE_CALL(cudaThreadSynchronize());
      timer.stop();
      CUDA_SAFE_CALL(cudaMemcpy(&cpuLock,     gpuLock,    sizeof(unsigned int), cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy(&cpuCounter,  gpuCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));

      if (cpuCounter == GRID_SIZES[i] * BLOCK_SIZES[j] && cpuLock == 0)
      {
        printf(" %9.3f ms    ", timer.getElapsedMilliseconds());
      }
      else
      {
        printf(" cnt=%-8u    ", cpuCounter);
      }
    }
    printf("\n");
  }
  printf("\n");

  printf("  gs/bs");
  for (int i = 0; i < NUM_BLOCK_SIZES_2; ++i)
  {
    printf("  | %13d", BLOCK_SIZES_2[i]);
  }
  printf("\n");
  printf("--------");
  for (int i = 0; i < NUM_BLOCK_SIZES_2; ++i) printf("-+---------------");
  printf("\n");
  for (int i = 0; i < NUM_GRID_SIZES; ++i)
  {
    printf("%7d  | ", GRID_SIZES[i]); fflush(stdout);
    dim3 gs(GRID_SIZES[i], 1, 1);
    for (int j = 0; j < NUM_BLOCK_SIZES_2; ++j)
    {
      CUDA_SAFE_CALL(cudaMemset(gpuLock,    0, sizeof(unsigned int)));
      CUDA_SAFE_CALL(cudaMemset(gpuCounter, 0, sizeof(unsigned int)));
      dim3 bs(BLOCK_SIZES_2[j], 1, 1);
      Timer timer;
      timer.start();
      kernel2<<<gs, bs>>>();
      CUDA_SAFE_CALL(cudaThreadSynchronize());
      timer.stop();
      CUDA_SAFE_CALL(cudaMemcpy(&cpuLock,     gpuLock,    sizeof(unsigned int), cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy(&cpuCounter,  gpuCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));

      if (cpuCounter == (GRID_SIZES[i] * BLOCK_SIZES_2[j] / 32) && cpuLock == 0)
      {
        printf(" %9.3f ms    ", timer.getElapsedMilliseconds());
      }
      else
      {
        printf(" cnt=%-8u    ", cpuCounter);
      }
      fflush(stdout);
    }
    printf("\n"); fflush(stdout);
  }
  printf("\n"); fflush(stdout);

  return 0;
}
