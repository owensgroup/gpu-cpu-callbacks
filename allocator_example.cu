#include <cudaSafeCall.h>
#include <cudaParams.h>
#include <cudaParamCreate.h>
#include <cudaCallFunc.h>
#include <cstdio>
#include <cmath>
#include <Timer.h>
#include <pthread.h>

#ifndef _WIN32
  #define __cdecl
#endif

Timer timerr, timerw;

const int NUM_THREADS = 32;
const int NUM_BLOCKS  = 4096;
const int INTS_PER_BLOCK = 256 * NUM_THREADS;
const int NUM_ACTIVE_PAGES = 240;

int mostActivePages = -1;
int nextActivePage = 0;
int gpuPages[NUM_ACTIVE_PAGES * 2];
pthread_mutex_t mutex;

extern "C"
{
  void __cdecl mallocMarshall(void * retPtr, void * params[])
  {
    pthread_mutex_lock(&mutex);
    printf("incrementing nextActivePage (currently %d)\n", nextActivePage); fflush(stdout);
    // printf("giving away %x\n", gpuPages[nextActivePage]); fflush(stdout);
    *reinterpret_cast<int * >(retPtr) = gpuPages[nextActivePage++];
    if (nextActivePage > mostActivePages) mostActivePages = nextActivePage;
    if (nextActivePage > NUM_ACTIVE_PAGES) { printf("error, too many pages requested, returning bogus data.\n"); fflush(stdout); }
    pthread_mutex_unlock(&mutex);
  }
  void __cdecl freeMarshall(void * retPtr, void * params[])
  {
    // printf("giving back %x\n", *reinterpret_cast<int * >(params[0]));
    pthread_mutex_lock(&mutex);
    printf("decrementing nextActivePage (currently %d)\n", nextActivePage); fflush(stdout);
    gpuPages[--nextActivePage] = *reinterpret_cast<int * >(params[0]);
    pthread_mutex_unlock(&mutex);
  }
}

__shared__ int offset;
__shared__ int usedToKeepActiveBlockNumberLow[3000];

__global__ void kernel(cudaCallbackData * cdata, cudaHostFunction_t mallocFunc, cudaHostFunction_t freeFunc, clock_t * clocks, int * memPool)
{
  if (threadIdx.x == 0)
  {
    clocks[blockIdx.x * 4 + 0] = clock();
    offset = cudaCallFunc<int>(cdata, mallocFunc);
    clocks[blockIdx.x * 4 + 1] = clock();
  }
  __syncthreads();
  int * t = memPool + offset;
  // do something to keep the PCI-e bus pressure down.
  for (int i = threadIdx.x; i < INTS_PER_BLOCK; i += blockDim.x) t[i] = i;
  __syncthreads();
  if (threadIdx.x == 0)
  {
    clocks[blockIdx.x * 4 + 2] = clock();
    cudaCallFunc<int>(cdata, freeFunc, offset);
    clocks[blockIdx.x * 4 + 3] = clock();
  }
}
__global__ void kernel2(clock_t * clocks)
{
  if (threadIdx.x == 0)
  {
    clocks[blockIdx.x * 4 + 0] = clock();
    clocks[blockIdx.x * 4 + 1] = clock();
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    clocks[blockIdx.x * 4 + 2] = clock();
    clocks[blockIdx.x * 4 + 3] = clock();
  }
}

int dcmp(const void * a, const void * b)
{
  const double * d0 = reinterpret_cast<const double * >(a);
  const double * d1 = reinterpret_cast<const double * >(b);
  if (*d0 < *d1) return -1;
  if (*d0 > *d1) return  1;
  return 0;
}

double clockDiff(const clock_t c1, const clock_t c2)
{
  unsigned long long int t1 = static_cast<unsigned long long int>(c1);
  unsigned long long int t2 = static_cast<unsigned long long int>(c2);
  int i1 = static_cast<int>(t1 & 0xFFFFFFFFUL);
  int i2 = static_cast<int>(t2 & 0xFFFFFFFFUL);

  if (i1 > i2)
  {
    i2 = static_cast<int>(t2 & 0x1FFFFFFFFULL);
  }

  return i2 - i1;
}

int main(int argc, char ** argv)
{
  const double GPU_CLOCK_FREQ = 1512000;
  dim3 gs = dim3(NUM_BLOCKS, 1, 1);
  dim3 bs = dim3(NUM_THREADS, 1, 1);

  cudaParamType_t params[4];
  cudaHostFunction_t mallocFunc, freeFunc;
  cudaCallbackData_t * callbackData;
  int * cpuMem, * gpuMem;
  clock_t * cpuClocks, * gpuClocks;

  CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_SAFE_CALL(cudaSetDevice(0));
  CUDA_SAFE_CALL(cudaHostCallbackInit(&callbackData));

  CUDA_SAFE_CALL(cudaHostAlloc           (reinterpret_cast<void ** >(&cpuClocks), sizeof(clock_t) * NUM_BLOCKS * 4,             cudaHostAllocMapped));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&gpuClocks), cpuClocks,                                    0));
  CUDA_SAFE_CALL(cudaHostAlloc           (reinterpret_cast<void ** >(&cpuMem), NUM_ACTIVE_PAGES * INTS_PER_BLOCK * sizeof(int), cudaHostAllocMapped));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&gpuMem), cpuMem,                                          0));

  printf("cpuMem gpuMem { %p %p }\n", cpuMem, gpuMem);

  pthread_mutex_init(&mutex, NULL);
  for (int i = 0; i < NUM_ACTIVE_PAGES; ++i)
  {
    gpuPages[i] = INTS_PER_BLOCK * i;
  }
  for (int i = NUM_ACTIVE_PAGES; i < NUM_ACTIVE_PAGES * 2; ++i)
  {
    gpuPages[i] = 0xFFFFFFFF;
  }

  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 0));
  CUDA_SAFE_CALL(cudaCreateHostFunc(&mallocFunc, (void * )mallocMarshall, params[0], 0, NULL));

  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 0));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 1));
  CUDA_SAFE_CALL(cudaCreateHostFunc(&freeFunc, (void * )freeMarshall, params[0], 1, params + 1));

  kernel<<<gs, bs, 0, 0>>>(callbackData, mallocFunc, freeFunc, gpuClocks, gpuMem);
  // kernel2<<<gs, bs, 0, 0>>>(gpuClocks);
  CUDA_SAFE_CALL(cudaGetLastError());
  CUDA_SAFE_CALL(cudaCallbackSynchronize(0));

  double * mallocDiffs = new double[NUM_BLOCKS];
  double * freeDiffs   = new double[NUM_BLOCKS];
  double mallocTotal   = 0.0, mallocStddev = 0.0, mallocMean = 0.0;
  double freeTotal     = 0.0, freeStddev   = 0.0, freeMean   = 0.0;
  for (int i = 0; i < NUM_BLOCKS; ++i)
  {
    mallocDiffs[i] = clockDiff(cpuClocks[i * 4 + 0], cpuClocks[i * 4 + 1]) / (double)GPU_CLOCK_FREQ;
    freeDiffs  [i] = clockDiff(cpuClocks[i * 4 + 2], cpuClocks[i * 4 + 3]) / (double)GPU_CLOCK_FREQ;
    mallocTotal += mallocDiffs[i];
    freeTotal   += freeDiffs  [i];
  }
  mallocMean = mallocTotal / NUM_BLOCKS;
  freeMean   = freeTotal   / NUM_BLOCKS;

  qsort(mallocDiffs, NUM_BLOCKS, sizeof(double), dcmp);
  qsort(freeDiffs,   NUM_BLOCKS, sizeof(double), dcmp);
  for (int i = 0; i < NUM_BLOCKS; ++i)
  {
    mallocStddev += (mallocDiffs[i] - mallocMean) * (mallocDiffs[i] - mallocMean);
    freeStddev   += (freeDiffs  [i] - freeMean)   * (freeDiffs  [i] - freeMean);
  }
  mallocStddev = sqrt(mallocStddev / NUM_BLOCKS);
  freeStddev   = sqrt(freeStddev   / NUM_BLOCKS);
  printf("mostActivePages: %d\n", mostActivePages);
  printf("%d mallocs { min med mean max stddev } { %.3f %.3f %.3f %.3f %.3f } ms\n", NUM_BLOCKS, mallocDiffs[0], mallocDiffs[NUM_BLOCKS / 2], mallocMean, mallocDiffs[NUM_BLOCKS - 1], mallocStddev);
  printf("%d frees   { min med mean max stddev } { %.3f %.3f %.3f %.3f %.3f } ms\n", NUM_BLOCKS, freeDiffs  [0], freeDiffs  [NUM_BLOCKS / 2], freeMean,   freeDiffs  [NUM_BLOCKS - 1], freeStddev);
  fflush(stdout);

  return 0;
}
