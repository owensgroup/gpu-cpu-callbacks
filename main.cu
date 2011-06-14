#include <cudaSafeCall.h>
#include <cudaParams.h>
#include <cudaParamCreate.h>
#include <cudaCallFunc.h>
#include <cstdio>
#include <cerrno>
#include <Timer.h>
#include <fcntl.h>
#include <sys/stat.h>

#ifdef _WIN32
  #include <io.h>
  #define open _open
  #define read _read
  #define write _write
#else
  #include <unistd.h>
  #include <fcntl.h>
  #define __cdecl
#endif

Timer timerr, timerw;

extern "C"
{
  void __cdecl readMarshall(void * retPtr, void * params[])
  {
    int     fd  = *reinterpret_cast<int *     >(params[0]);
    void *  mem = *reinterpret_cast<void **   >(params[1]);
    size_t size = *reinterpret_cast<size_t *  >(params[2]);

    bool keep = true;
    int ret = 0;
    timerr.start();
    while (ret < size && keep)
    {
      int t = read(fd, mem, size);
      switch (t)
      {
      case -1:
        ret = -1;
        keep = false;
        break;
      case 0:
        keep = false;
        break;
      default:
        ret += t;
        break;
      }
    }
    timerr.stop();

    // fprintf(stderr, "read(fd=%d, mem=%p, size=%u)=%d\n", fd, mem, size, ret); fflush(stderr);
    if (ret < 0)
    {
      printf("strerror(%d): %s\n", errno, strerror(errno));
    }
    ret = *reinterpret_cast<int * >(retPtr);
  }
  void __cdecl writeMarshall(void * retPtr, void * params[])
  {
    int     fd  = *reinterpret_cast<int *     >(params[0]);
    void *  mem = *reinterpret_cast<void **   >(params[1]);
    size_t size = *reinterpret_cast<size_t *  >(params[2]);

    bool keep = true;
    int ret = 0;
    timerw.start();
    while (ret < size && keep)
    {
      int t = write(fd, mem, size);
      switch (t)
      {
      case -1:
        ret = -1;
        keep = false;
        break;
      case 0:
        keep = false;
        break;
      default:
        ret += t;
        break;
      }
    }
    timerw.stop();

    // fprintf(stderr, "write(fd=%d, mem=%p, size=%u)=%d\n", fd, mem, size, ret); fflush(stderr);
    if (ret <= 0)
    {
      printf("strerror(%d): %s\n", errno, strerror(errno));
    }
    *reinterpret_cast<int * >(retPtr) = ret;
  }
}

__global__ void kernel(cudaCallbackData * cdata, cudaHostFunction_t readFunc, cudaHostFunction_t writeFunc, int rfd, int wfd, size_t size, char * mem, char * cpuMem)
{
  clock_t * start1, * end1, * start2, * end2;

  start1 = reinterpret_cast<clock_t * >(mem); mem += sizeof(clock_t);
  end1   = reinterpret_cast<clock_t * >(mem); mem += sizeof(clock_t);
  start2 = reinterpret_cast<clock_t * >(mem); mem += sizeof(clock_t);
  end2   = reinterpret_cast<clock_t * >(mem); mem += sizeof(clock_t);
  cpuMem += sizeof(clock_t) * 4;

  *start1 = clock();
  cudaCallFunc<int>(cdata, readFunc,  rfd, cpuMem, size);
  *end1   = clock();

  *start2 = clock();
  cudaCallFunc<int>(cdata, writeFunc, wfd, cpuMem, size);
  *end2   = clock();
}

int main(int argc, char ** argv)
{
  const double GPU_CLOCK_FREQ = 1512000;
  if (argc != 4)
  {
    fprintf(stderr, "Usage: %s input_file output_file size\n", argv[0]);
    fflush(stderr);
    return 0;
  }
  dim3 gs = dim3(1, 1, 1);
  dim3 bs = dim3(1, 1, 1);

  size_t size;
  int rfd = -1, wfd = -1;
  cudaParamType_t params[4];
  cudaHostFunction_t readFunc, writeFunc;
  cudaCallbackData_t * callbackData;
  char * cpuMem, * gpuMem;

  sscanf(argv[3], "%lu", &size);

  rfd = open(argv[1], O_RDONLY);
  wfd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0644);

  if (rfd == -1)
  {
    fprintf(stderr, "Couldn't open %s for reading.\n", argv[1]);
    return 1;
  }
  if (wfd == -1)
  {
    fprintf(stderr, "Couldn't open %s for writing.\n", argv[2]);
    return 1;
  }

  CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_SAFE_CALL(cudaSetDevice(0));
  CUDA_SAFE_CALL(cudaHostCallbackInit(&callbackData));

  CUDA_SAFE_CALL(cudaHostAlloc(reinterpret_cast<void ** >(&cpuMem), size + sizeof(clock_t) * 4, cudaHostAllocMapped));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&gpuMem), cpuMem, 0));

  CUDA_SAFE_CALL(cudaParamCreate<int    >(params + 0));
  CUDA_SAFE_CALL(cudaParamCreate<int    >(params + 1));
  CUDA_SAFE_CALL(cudaParamCreate<char * >(params + 2));
  CUDA_SAFE_CALL(cudaParamCreate<size_t >(params + 3));
  CUDA_SAFE_CALL(cudaCreateHostFunc(&readFunc, (void * )readMarshall, params[0], 3, params + 1));

  CUDA_SAFE_CALL(cudaParamCreate<int    >(params + 0));
  CUDA_SAFE_CALL(cudaParamCreate<int    >(params + 1));
  CUDA_SAFE_CALL(cudaParamCreate<char * >(params + 2));
  CUDA_SAFE_CALL(cudaParamCreate<size_t >(params + 3));
  CUDA_SAFE_CALL(cudaCreateHostFunc(&writeFunc, (void * )writeMarshall, params[0], 3, params + 1));

  cudaStream_t kernelStream;
  CUDA_SAFE_CALL(cudaStreamCreate(&kernelStream));

  kernel<<<gs, bs, 0, kernelStream>>>(callbackData, readFunc, writeFunc, rfd, wfd, size, reinterpret_cast<char * >(gpuMem), reinterpret_cast<char * >(cpuMem));
  CUDA_SAFE_CALL(cudaGetLastError());
  CUDA_SAFE_CALL(cudaCallbackSynchronize(kernelStream));

  clock_t start1, end1, start2, end2;
  char * mem = cpuMem;

  start1 = *reinterpret_cast<clock_t * >(mem); mem += sizeof(clock_t);
  end1   = *reinterpret_cast<clock_t * >(mem); mem += sizeof(clock_t);
  start2 = *reinterpret_cast<clock_t * >(mem); mem += sizeof(clock_t);
  end2   = *reinterpret_cast<clock_t * >(mem); mem += sizeof(clock_t);

  printf("%10lu -", size);
  printf(" %20.6f ms", timerr.getElapsedMilliseconds());
  printf(" %20.6f ms", timerw.getElapsedMilliseconds());
  printf(" %20.6f ms", (double)(end1 - start1) / GPU_CLOCK_FREQ);
  printf(" %20.6f ms", (double)(end2 - start2) / GPU_CLOCK_FREQ);
  printf("\n");

  return 0;
}
