#include <cudaSafeCall.h>
#include <cudaParams.h>
#include <cudaParamCreate.h>
#include <cudaCallFunc.h>
#include <cstdio>
#include <cerrno>
#include <Timer.h>
#include <ThreadStuff.h>

#if 0
  const int IMAGE_DIM         = 1024;
  const int NUM_KERNEL_RUNS   = 15;
  const int NUM_IMAGES        = 30;
  const int NUM_REPS          = 20;
#else
  const int IMAGE_DIM         = 2048;
  const int NUM_KERNEL_RUNS   =  3;
  const int NUM_IMAGES        =  4;
  const int NUM_REPS          = 80;
#endif

const int FILTER_BUFFER     = 2;
const int FILTER_ROWS       = FILTER_BUFFER * 2 + 1;
const int FILTER_COLS       = FILTER_BUFFER * 2 + 1;
const int INNER_DIM         = 16;
const int BLOCKS_PER_IMAGE  = (IMAGE_DIM / INNER_DIM) * (IMAGE_DIM / INNER_DIM);
const int NUM_SAMPLES       = (FILTER_BUFFER * 2 + INNER_DIM) * (FILTER_BUFFER * 2 + INNER_DIM);
const int NUM_INNER_SAMPLES = INNER_DIM * INNER_DIM;

volatile bool readInProgress    = false;
volatile bool writeInProgress   = false;
volatile int kernelFinished     =  0;
volatile int memcpyDownFinished = -1;
volatile int memcpyUpFinished   = -1;

class ReadImageRunner;
class WriteImageRunner;

Lock * readLock = NULL;
Lock * writeLock = NULL;
Condition * readCond = NULL;
Condition * writeCond = NULL;
Thread * readThread = NULL;
Thread * writeThread = NULL;
ReadImageRunner * readRunner = NULL;
WriteImageRunner * writeRunner = NULL;

typedef struct _Image
{
  float pixels[IMAGE_DIM][IMAGE_DIM];
} Image;

__shared__ float  KERNEL[FILTER_ROWS][FILTER_COLS];
__shared__ float  samples[FILTER_ROWS + INNER_DIM][FILTER_COLS + INNER_DIM];

__device__ void moveInput(const Image * const input, const int startRow, const int startCol)
{
  for (int i = threadIdx.x; i < NUM_SAMPLES; i += blockDim.x)
  {
    const int r0 = i / (FILTER_BUFFER * 2 + INNER_DIM);
    const int c0 = i % (FILTER_BUFFER * 2 + INNER_DIM);
    const int r1 = startRow - FILTER_BUFFER + r0;
    const int c1 = startCol - FILTER_BUFFER + c0;
    const int row = (r1 < 0) ? 0 : (r1 >= IMAGE_DIM) ? IMAGE_DIM - 1 : r1;
    const int col = (c1 < 0) ? 0 : (c1 >= IMAGE_DIM) ? IMAGE_DIM - 1 : c1;
    samples[r0][c0] = input->pixels[row][col];
  }
}

__device__ void moveOutput(Image * const output, const int startRow, const int startCol)
{
  const int NUM_SAMPLES = INNER_DIM * INNER_DIM;
  for (int i = threadIdx.x; i < NUM_SAMPLES; i += blockDim.x)
  {
    const int r0  = i / INNER_DIM;
    const int c0  = i % INNER_DIM;
    const int row = startRow + r0;
    const int col = startCol + c0;
    output->pixels[row][col] = samples[r0 + FILTER_BUFFER][c0 + FILTER_BUFFER];
  }
}
/*
__global__ void callbackKernel(cudaCallbackData * cdata, cudaHostFunction_t finishFunc, float ** images, int * imageIDs)
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
*/

__global__ void regularKernel(const int numImages, Image * inImages, Image * outImages, const float * const kernel, const float kernelFactor)
{
  // if (blockIdx.x < 4096) return;

  const int TOTAL_BLOCKS = numImages * BLOCKS_PER_IMAGE;

  for (int index = threadIdx.x; index < FILTER_ROWS * FILTER_COLS; index += blockDim.x)
  {
    KERNEL[index / FILTER_COLS][index % FILTER_COLS] = kernel[index];
  }

  for (int i = blockIdx.x; i < TOTAL_BLOCKS; i += gridDim.x)
  {
    const int IMAGE_INDEX     = i / BLOCKS_PER_IMAGE;
    const int CHUNK_INDEX     = i % BLOCKS_PER_IMAGE;
    const int CHUNKS_PER_ROW  = (IMAGE_DIM / INNER_DIM);
    const int CHUNK_ROW_INDEX = CHUNK_INDEX / CHUNKS_PER_ROW;
    const int CHUNK_COL_INDEX = CHUNK_INDEX % CHUNKS_PER_ROW;
    const int ROW_INDEX       = CHUNK_ROW_INDEX * INNER_DIM;
    const int COL_INDEX       = CHUNK_COL_INDEX * INNER_DIM;
    moveInput(inImages + IMAGE_INDEX, ROW_INDEX, COL_INDEX);
    __syncthreads();

    for (int sampleIndex = threadIdx.x; sampleIndex < NUM_INNER_SAMPLES; sampleIndex += blockDim.x)
    {
      const int X = sampleIndex % INNER_DIM;
      const int Y = sampleIndex / INNER_DIM;
      float sum = 0.0f;
      for (int row = -FILTER_BUFFER; row <= FILTER_BUFFER; ++row)
      {
        for (int col = -FILTER_BUFFER; col <= FILTER_BUFFER; ++col)
        {
          sum += samples[FILTER_BUFFER + row + Y][FILTER_BUFFER + col + X] * KERNEL[row + FILTER_BUFFER][col + FILTER_BUFFER];
        }
      }
      outImages[IMAGE_INDEX].pixels[ROW_INDEX + Y][COL_INDEX + X] = sum * kernelFactor;
    }
  }
}

inline unsigned char clamp(const float f)
{
  return f < 0.0f ? 0 : f > 255.0f ? 255 : static_cast<unsigned char>(static_cast<unsigned int>(f));
}

class ReadImageRunner : public Runner
{
  public:
    Image * cpuImages;
    unsigned char * imageRaw;
    inline ReadImageRunner()
    {
    }
    inline virtual ~ReadImageRunner()
    {
    }
    inline void run()
    {
      imageRaw = new unsigned char[IMAGE_DIM * IMAGE_DIM];
      fprintf(stderr, "%s.ReadImageRunner::run.%d\n", __FILE__, __LINE__); fflush(stderr);
      for (int image = 0; image < NUM_IMAGES; ++image)
      {
        FILE * fp = fopen("1024.pgm", "rb");
        char buf[1024];
        fgets(buf, 1023, fp);
        fgets(buf, 1023, fp);
        fgets(buf, 1023, fp);
        fread(imageRaw, 1048576, 1, fp);
        fclose(fp);
        for (int i = 0; i < IMAGE_DIM; ++i)
        {
          for (int j = 0; j < IMAGE_DIM; ++j)
          {
            cpuImages[image].pixels[i][j] = static_cast<float>(imageRaw[i * IMAGE_DIM + j]) / 255.0f;
          }
        }
      }
      // fprintf(stderr, "%s.ReadImageRunner::run.%d\n", __FILE__, __LINE__); fflush(stderr);
      readLock->lock();
      // fprintf(stderr, "%s.ReadImageRunner::run.%d: setting readInProgress to false.\n", __FILE__, __LINE__); fflush(stderr);
      readInProgress = false;
      readCond->signal();
      readLock->unlock();
      // fprintf(stderr, "%s.ReadImageRunner::run.%d\n", __FILE__, __LINE__); fflush(stderr);
    }
};
class WriteImageRunner : public Runner
{
  public:
    Image * cpuOutImages;
    unsigned char * imageRaw;
    int rep;
    inline WriteImageRunner()
    {
    }
    inline virtual ~WriteImageRunner()
    {
    }
    inline virtual void run()
    {
      imageRaw = new unsigned char[IMAGE_DIM * IMAGE_DIM];
      fprintf(stderr, "%s.WriteImageRunner::run.%d", __FILE__, __LINE__); fflush(stderr);
      for (int image = 0; image < NUM_IMAGES; ++image)
      {
        char imgName[100];
        sprintf(imgName, "/tmp/image.%05d.%05d.pgm", rep, image);

        FILE * fp = fopen(imgName, "wb");
        fprintf(fp, "P5\n1024 1024\n255\n");

        for (int i = 0; i < IMAGE_DIM; ++i)
        {
          for (int j = 0; j < IMAGE_DIM; ++j)
          {
            imageRaw[i * IMAGE_DIM + j] = clamp(cpuOutImages[image].pixels[i][j] * 255.0f);
          }
        }
        fwrite(imageRaw, IMAGE_DIM * IMAGE_DIM, 1, fp);
        fclose(fp);
      }
      // fprintf(stderr, "%s.WriteImageRunner::run.%d\n", __FILE__, __LINE__); fflush(stderr);
      writeLock->lock();
      // fprintf(stderr, "%s.WriteImageRunner::run.%d: setting writeInProgress to false.\n", __FILE__, __LINE__); fflush(stderr);
      writeInProgress = false;
      writeCond->signal();
      writeLock->unlock();
      // fprintf(stderr, "%s.WriteImageRunner::run.%d\n", __FILE__, __LINE__); fflush(stderr);
    }
};

inline void readImages(Image * const cpuImages, unsigned char * const imageRaw)
{
  readLock->lock();
  fprintf(stderr, "%s.%s.%d: setting readInProgress to true.\n", __FILE__, __FUNCTION__, __LINE__); fflush(stderr);
  readInProgress = true;
  readLock->unlock();

  readRunner->cpuImages = cpuImages;
  readRunner->imageRaw  = imageRaw;
  readThread->start();
}
inline void writeImages(Image * const cpuOutImages, unsigned char * const imageRaw, const int rep)
{
  writeLock->lock();
  // fprintf(stderr, "%s.%s.%d: setting writeInProgress to true.\n", __FILE__, __FUNCTION__, __LINE__); fflush(stderr);
  writeInProgress = true;
  writeLock->unlock();

  writeRunner->cpuOutImages = cpuOutImages;
  writeRunner->imageRaw     = imageRaw;
  writeRunner->rep          = rep;
  writeThread->start();
}

inline void waitForRead()
{
  readLock->lock();
  // fprintf(stderr, "%s.%s.%d: waiting for read.\n", __FILE__, __FUNCTION__, __LINE__); fflush(stderr);
  if (readInProgress) readCond->wait();
  readThread->join();
  // fprintf(stderr, "%s.%s.%d: done waiting for read.\n", __FILE__, __FUNCTION__, __LINE__); fflush(stderr);
  readLock->unlock();
}

inline void waitForWrite()
{
  writeLock->lock();
  // fprintf(stderr, "%s.%s.%d: waiting for write.\n", __FILE__, __FUNCTION__, __LINE__); fflush(stderr);
  if (writeInProgress) writeCond->wait();
  writeThread->join();
  // fprintf(stderr, "%s.%s.%d: done waiting for write.\n", __FILE__, __FUNCTION__, __LINE__); fflush(stderr);
  writeLock->unlock();
}

inline void scheduleKernels(Image * gpuImages, Image * gpuOutImages, float * const gpuKernel, const float kernelFactor)
{
  dim3 gridSize((IMAGE_DIM * IMAGE_DIM) / (INNER_DIM * INNER_DIM), 1, 1);
  dim3 blockSize(INNER_DIM * INNER_DIM, 1, 1);

  for (int i = 0; i < NUM_KERNEL_RUNS; ++i)
  {
    regularKernel<<<gridSize, blockSize>>>(NUM_IMAGES, gpuImages, gpuOutImages, gpuKernel, kernelFactor);
    Image * t = gpuImages;
    gpuImages = gpuOutImages;
    gpuOutImages = t;
  }
}

inline int getIndex(const int ind)
{
  return (ind + 3) % 3;
}

int main(int argc, char ** argv)
{
  Timer t0;
  FILE * fp;
  float kernel[FILTER_ROWS][FILTER_COLS] =
  {
    { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, },
    { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, },
    { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, },
    { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, },
    { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, },
  };
  const float kernelFactor = 1.0f / 25.0f;
  float * cpuKernel, * gpuKernel;
  Image * cpuInImages0,  * cpuInImages1,  * cpuInImages2,  * gpuInImages0,  * gpuInImages1,  * gpuInImages2;
  Image * cpuOutImages0, * cpuOutImages1, * cpuOutImages2, * gpuOutImages0, * gpuOutImages1, * gpuOutImages2;
  unsigned char * imageRawRead  = new unsigned char[IMAGE_DIM * IMAGE_DIM];
  unsigned char * imageRawWrite = new unsigned char[IMAGE_DIM * IMAGE_DIM];

  readRunner  = new ReadImageRunner();
  writeRunner = new WriteImageRunner();
  readThread  = new Thread(readRunner);
  writeThread = new Thread(writeRunner);
  readLock    = new Lock();
  writeLock   = new Lock();
  readCond    = readLock->newCondition();
  writeCond   = writeLock->newCondition();

  CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_SAFE_CALL(cudaSetDevice(1));
  CUDA_SAFE_CALL(cudaHostAlloc           (reinterpret_cast<void ** >(&cpuKernel),      sizeof(kernel)            , cudaHostAllocMapped | cudaHostAllocPortable));
  CUDA_SAFE_CALL(cudaHostAlloc           (reinterpret_cast<void ** >(&cpuInImages0),   sizeof(Image) * NUM_IMAGES, cudaHostAllocMapped | cudaHostAllocPortable));
  CUDA_SAFE_CALL(cudaHostAlloc           (reinterpret_cast<void ** >(&cpuInImages1),   sizeof(Image) * NUM_IMAGES, cudaHostAllocMapped | cudaHostAllocPortable));
  CUDA_SAFE_CALL(cudaHostAlloc           (reinterpret_cast<void ** >(&cpuInImages2),   sizeof(Image) * NUM_IMAGES, cudaHostAllocMapped | cudaHostAllocPortable));
  CUDA_SAFE_CALL(cudaHostAlloc           (reinterpret_cast<void ** >(&cpuOutImages0),  sizeof(Image) * NUM_IMAGES, cudaHostAllocMapped | cudaHostAllocPortable));
  CUDA_SAFE_CALL(cudaHostAlloc           (reinterpret_cast<void ** >(&cpuOutImages1),  sizeof(Image) * NUM_IMAGES, cudaHostAllocMapped | cudaHostAllocPortable));
  CUDA_SAFE_CALL(cudaHostAlloc           (reinterpret_cast<void ** >(&cpuOutImages2),  sizeof(Image) * NUM_IMAGES, cudaHostAllocMapped | cudaHostAllocPortable));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&gpuKernel),      cpuKernel                 , 0));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&gpuInImages0),   cpuInImages0              , 0));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&gpuInImages1),   cpuInImages1              , 0));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&gpuInImages2),   cpuInImages1              , 0));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&gpuOutImages0),  cpuOutImages0             , 0));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&gpuOutImages1),  cpuOutImages1             , 0));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&gpuOutImages2),  cpuOutImages2             , 0));

  Image * cpuInImages [] = { cpuInImages0,  cpuInImages1,  cpuInImages2  };
  Image * cpuOutImages[] = { cpuOutImages0, cpuOutImages1, cpuOutImages2 };
  Image * gpuInImages [] = { gpuInImages0,  gpuInImages1,  gpuInImages2  };
  Image * gpuOutImages[] = { gpuOutImages0, gpuOutImages1, gpuOutImages2 };

  memcpy(cpuKernel, kernel, sizeof(kernel));

  t0.start();

  {
    unsigned char * imageRaw = new unsigned char[IMAGE_DIM * IMAGE_DIM];
    for (int image = 0; image < 1; ++image)
    {
      FILE * fp = fopen("1024.pgm", "rb");
      char buf[1024];
      fgets(buf, 1023, fp);
      fgets(buf, 1023, fp);
      fgets(buf, 1023, fp);
      fread(imageRaw, 1048576, 1, fp);
      fclose(fp);
      for (int i = 0; i < IMAGE_DIM; ++i)
      {
        for (int j = 0; j < IMAGE_DIM; ++j)
        {
          cpuInImages0[image].pixels[i][j] = static_cast<float>(imageRaw[i * IMAGE_DIM + j]) / 255.0f;
        }
      }
    }
    delete [] imageRaw;
  }

  t0.stop();

  t0.start();

  readImages(cpuInImages[getIndex(0)], imageRawRead);
  for (int rep = 0; rep < NUM_REPS; ++rep)
  {
    if (rep != 0)
    {
      waitForWrite();
      CUDA_SAFE_CALL(cudaThreadSynchronize());
      writeImages(cpuOutImages[getIndex(rep - 1)], imageRawWrite, rep);
    }
    waitForRead();
    scheduleKernels(gpuInImages[getIndex(rep)], gpuOutImages[getIndex(rep)], gpuKernel, kernelFactor);
    if (rep != NUM_REPS - 1) readImages(cpuInImages[getIndex(rep + 1)], imageRawRead);
  }
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  waitForRead();

  writeImages(cpuOutImages[getIndex(NUM_REPS - 1)], imageRawWrite, NUM_REPS);

  waitForWrite();
  t0.stop();

  for (int i = 0; i < IMAGE_DIM; ++i)
  {
    for (int j = 0; j < IMAGE_DIM; ++j)
    {
      imageRawWrite[i * IMAGE_DIM + j] = clamp(cpuOutImages0[0].pixels[i][j] * 255.0f);
    }
  }

  fp = fopen("1024.out.pgm", "wb");
  fprintf(fp, "P5\n1024 1024\n255\n");
  fwrite(imageRawWrite, 1048576, 1, fp);
  fclose(fp);

  delete [] imageRawRead;
  delete [] imageRawWrite;

  printf("%d images took %.3f ms \n", NUM_IMAGES * NUM_REPS, t0.getElapsedMilliseconds());

  return 0;
}
