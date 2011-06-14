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
const int QUEUE_CAPACITY = 2048;
const int INITIAL_QUEUE_SIZE = 1024;
const int TOTAL_ELEMENTS_TO_PROCESS = 1024;

int mostActivePages = -1;
int nextActivePage = 0;
bool blockIsWaiting[NUM_BLOCKS] = { false };
int gpuPages[NUM_ACTIVE_PAGES * 2];
int cpuQueue[QUEUE_CAPACITY], queueFront, queueBack, queueTotal;
pthread_mutex_t mutex;
int * cpuFlags, * gpuFlags;
int numWorking = 0;
int numWaiting = 0;
int numDone    = 0;
int numIdle    = NUM_BLOCKS;

enum
{
  STATUS_WORK_LEFT = 0,
  STATUS_WAITING,
  STATUS_DONE,
};

extern "C"
{
  void addToQueue(const int val)
  {
    cpuQueue[queueBack] = val;
    queueBack = (queueBack + 1) % QUEUE_CAPACITY;
  }
  int removeFromQueue()
  {
    int ret = cpuQueue[queueFront];
    queueFront = (queueFront + 1) % QUEUE_CAPACITY;
    queueTotal++;
    return ret;
  }
  void __cdecl getMarshall(void * retPtr, void * params[])
  {
    pthread_mutex_lock(&mutex);
    // printf("incrementing nextActivePage (currently %d)\n", nextActivePage); fflush(stdout);
    // printf("giving away %x\n", gpuPages[nextActivePage]); fflush(stdout);
    int blockIndex = *reinterpret_cast<int * >(params[0]);
    if (queueTotal + numWorking < TOTAL_ELEMENTS_TO_PROCESS)
    {
      // printf("incrememting nextActivePage (currently %d) for block %d with %d working right now.\n", nextActivePage, blockIndex, numWorking); fflush(stdout);
      printf("telling block %5d to work - working waiting done idle { %5d %5d %5d %5d }.\n", blockIndex, numWorking, numWaiting, numDone, numIdle); fflush(stdout);
      cpuFlags[blockIndex * 2 + 0] = gpuPages[nextActivePage++];
      cpuFlags[blockIndex * 2 + 1] = removeFromQueue();
      *reinterpret_cast<int * >(retPtr) = STATUS_WORK_LEFT;
      ++numWorking;
      --numIdle;
    }
    else if (queueTotal >= TOTAL_ELEMENTS_TO_PROCESS)
    {
      printf("telling block %5d to quit - working waiting done idle { %5d %5d %5d %5d }.\n", blockIndex, numWorking, numWaiting, numDone, numIdle); fflush(stdout);
      *reinterpret_cast<int * >(retPtr) = STATUS_DONE;
      ++numDone;
      if (blockIsWaiting[blockIndex])
      {
        --numWaiting;
        blockIsWaiting[blockIndex] = false;
      }
    }
    else                                              
    {
      printf("telling block %5d to wait - working waiting done idle { %5d %5d %5d %5d }.\n", blockIndex, numWorking, numWaiting, numDone, numIdle); fflush(stdout);
      *reinterpret_cast<int * >(retPtr) = STATUS_WAITING;
      if (!blockIsWaiting[blockIndex])
      {
        ++numWaiting;
        blockIsWaiting[blockIndex] = true;
      }
    }
    if (nextActivePage > NUM_ACTIVE_PAGES) { printf("error, too many pages requested, returning bogus data.\n"); fflush(stdout); }
    pthread_mutex_unlock(&mutex);
  }
  void __cdecl doneMarshall(void * retPtr, void * params[])
  {
    // printf("giving back %x\n", *reinterpret_cast<int * >(params[0]));
    pthread_mutex_lock(&mutex);
    // printf("decrementing nextActivePage (currently %d)\n", nextActivePage); fflush(stdout);
    gpuPages[--nextActivePage] = *reinterpret_cast<int * >(params[0]);
    // addToQueue(*reinterpret_cast<int * >(params[1]));
    addToQueue(queueTotal + INITIAL_QUEUE_SIZE);
    printf("adding %d to queue with %d working right now.\n", queueTotal + INITIAL_QUEUE_SIZE, numWorking); fflush(stdout);
    --numWorking;
    ++numIdle;
    pthread_mutex_unlock(&mutex);
  }
}

__shared__ int usedToKeepActiveBlockNumberLow[3000];

/* LOGIC:

  CPU has an initial queue of 1024 items. GPU blocks chew through items. As each
  block finishes, it frees a chunk of memory it was using. If the CPU hasn't had
  a total of 128k elements processed in the queue, then it also adds a new
  element to the queue. If the CPU has gone through that many elements, then it
  tells the GPU to finish up.

*/

__global__ void kernel(cudaCallbackData * cdata,
                       cudaHostFunction_t getFunc,
                       cudaHostFunction_t doneFunc,
                       int * memPool,
                       int * flags)
{
  __shared__ int status;
  __shared__ int offset;
  __shared__ int multiplier;

  int loopsSoFar = 0;

  do
  {
    if (threadIdx.x == 0)
    {
      status      = cudaCallFunc<int>(cdata, getFunc, blockIdx.x);
      offset      = flags[blockIdx.x * 2 + 0];
      multiplier  = flags[blockIdx.x * 2 + 1];
    }
    __syncthreads();
    if (status == STATUS_WORK_LEFT)
    {
      // work is really simple, just do some arithmetic.
      int * base = memPool + INTS_PER_BLOCK * offset;
      for (int i = 0; i < INTS_PER_BLOCK; i += blockDim.x)
      {
        int res  = (threadIdx.x << 1) * (multiplier << 1) * (i << 1);
        res     += (threadIdx.x << 1) * (multiplier << 1) * (i >> 1);
        res     += (threadIdx.x << 1) * (multiplier >> 1) * (i << 1);
        res     += (threadIdx.x << 1) * (multiplier >> 1) * (i >> 1);
        res     += (threadIdx.x >> 1) * (multiplier << 1) * (i << 1);
        res     += (threadIdx.x >> 1) * (multiplier << 1) * (i >> 1);
        res     += (threadIdx.x >> 1) * (multiplier >> 1) * (i << 1);
        res     += (threadIdx.x >> 1) * (multiplier >> 1) * (i >> 1);
        base[i + threadIdx.x] = res;
      }
      __syncthreads();
      if (threadIdx.x == 0)
      {
        ++loopsSoFar;
        cudaCallFunc<int>(cdata, doneFunc, offset, loopsSoFar * blockIdx.x);
      }
      __syncthreads();
    }
    __syncthreads();
  }
  while (status != STATUS_DONE);
  __syncthreads();
}

int main(int argc, char ** argv)
{
  dim3 gs = dim3(NUM_BLOCKS, 1, 1);
  dim3 bs = dim3(NUM_THREADS, 1, 1);

  cudaParamType_t params[3];
  cudaHostFunction_t getFunc, doneFunc;
  cudaCallbackData_t * callbackData;
  int * cpuMem, * gpuMem;

  CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_SAFE_CALL(cudaSetDevice(0));
  CUDA_SAFE_CALL(cudaHostCallbackInit(&callbackData));

  CUDA_SAFE_CALL(cudaHostAlloc           (reinterpret_cast<void ** >(&cpuMem), NUM_ACTIVE_PAGES * INTS_PER_BLOCK * sizeof(int), cudaHostAllocMapped));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&gpuMem), cpuMem,                                          0));

  CUDA_SAFE_CALL(cudaHostAlloc           (reinterpret_cast<void ** >(&cpuFlags), NUM_BLOCKS * 2 * sizeof(int), cudaHostAllocMapped));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&gpuFlags), cpuMem,                       0));

  pthread_mutex_init(&mutex, NULL);
  for (int i = 0; i < NUM_ACTIVE_PAGES; ++i)
  {
    gpuPages[i] = INTS_PER_BLOCK * i;
  }
  for (int i = NUM_ACTIVE_PAGES; i < NUM_ACTIVE_PAGES * 2; ++i)
  {
    gpuPages[i] = 0xFFFFFFFF;
  }
  queueFront = queueBack = queueTotal = 0;
  for (int i = 0; i < INITIAL_QUEUE_SIZE; ++i) addToQueue(i);

  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 0));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 1));
  CUDA_SAFE_CALL(cudaCreateHostFunc(&getFunc, (void * )getMarshall, params[0], 1, params + 1));

  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 0));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 1));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 2));
  CUDA_SAFE_CALL(cudaCreateHostFunc(&doneFunc, (void * )doneMarshall, params[0], 2, params + 1));

  kernel<<<gs, bs, 0, 0>>>(callbackData, getFunc, doneFunc, gpuMem, gpuFlags);
  // kernel2<<<gs, bs, 0, 0>>>(gpuClocks);
  CUDA_SAFE_CALL(cudaGetLastError());
  CUDA_SAFE_CALL(cudaCallbackSynchronize(0));

  printf("mostActivePages: %d\n", mostActivePages);
  fflush(stdout);

  return 0;
}
