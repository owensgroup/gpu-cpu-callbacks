#include <cudaSafeCall.h>
#include <cudaParams.h>
#include <cudaParamCreate.h>
#include <cudaCallFunc.h>
#include <cstdio>
#include <cerrno>
#include <Timer.h>
#include <ThreadStuff.h>

#if 0
  const int IMAGE_DIM             = 1024;
  const int NUM_PASSES_PER_IMAGE  = 15;
  const int NUM_IMAGES            = 30;
  const int TOTAL_IMAGES          = 20;
#else
  const int IMAGE_DIM = 1024;
  int NUM_PASSES_PER_IMAGE;
  int NUM_IMAGES_IN_CORE;
  int TOTAL_IMAGES;
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


typedef struct _Image
{
  float pixels[IMAGE_DIM][IMAGE_DIM];
} Image;

Image * imageTemplate;

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
      for (int image = 0; image < NUM_IMAGES_IN_CORE; ++image)
      {
/*        FILE * fp = fopen("1024.pgm", "rb");
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
        */
        memcpy(cpuImages + image, imageTemplate, sizeof(*imageTemplate));
      }
      readLock->lock();
      printf("setting readInProgress to false.\n"); fflush(stdout);
      readInProgress = false;
      readCond->signal();
      readLock->unlock();
      delete [] imageRaw;
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
      for (int image = 0; image < NUM_IMAGES_IN_CORE; ++image)
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
      writeLock->lock();
      printf("setting writeInProgress to false.\n"); fflush(stdout);
      writeInProgress = false;
      writeCond->signal();
      writeLock->unlock();
      delete [] imageRaw;
    }
};

inline void readImages(Image * const cpuImages, unsigned char * const imageRaw)
{
  if (readThread->isRunning()) readThread->join();
  readLock->lock();
  printf("setting readInProgress to true.\n"); fflush(stdout);
  readInProgress = true;
  readLock->unlock();

  readRunner->cpuImages = cpuImages;
  readRunner->imageRaw  = imageRaw;
  readThread->start();
}
inline void writeImages(Image * const cpuOutImages, unsigned char * const imageRaw, const int rep)
{
  if (writeThread->isRunning()) writeThread->join();
  // writeLock->lock();
  // printf("setting writeInProgress to true.\n"); fflush(stdout);
  // writeInProgress = true;
  // writeLock->unlock();
  //
  // writeRunner->cpuOutImages = cpuOutImages;
  // writeRunner->imageRaw     = imageRaw;
  // writeRunner->rep          = rep;
  // writeThread->start();
}
inline void scheduleKernels(Image * gpuImages, Image * gpuOutImages, float * const gpuKernel, const float kernelFactor, cudaStream_t kernelStream, const int numImagesInCore)
{
  dim3 gridSize((IMAGE_DIM * IMAGE_DIM) / (INNER_DIM * INNER_DIM), 1, 1);
  dim3 blockSize(INNER_DIM * INNER_DIM, 1, 1);

  for (int i = 0; i < NUM_PASSES_PER_IMAGE; ++i)
  {
    regularKernel<<<gridSize, blockSize, 0, kernelStream>>>(numImagesInCore, gpuImages, gpuOutImages, gpuKernel, kernelFactor);
    Image * t = gpuImages;
    gpuImages = gpuOutImages;
    gpuOutImages = t;
  }
}

template <typename T>
inline void jswap(T & lhs, T & rhs)
{
  T t = lhs;
  lhs = rhs;
  rhs = t;
}

inline void waitForRead()
{
  readLock->lock();
  printf("%s.%s.%d: waiting for read.\n", __FILE__, __FUNCTION__, __LINE__); fflush(stdout);
  if (readInProgress) readCond->wait();
  readThread->join();
  printf("%s.%s.%d: done waiting for read.\n", __FILE__, __FUNCTION__, __LINE__); fflush(stdout);
  readLock->unlock();
}

inline void waitForWrite()
{
  writeLock->lock();
  printf("%s.%s.%d: waiting for write.\n", __FILE__, __FUNCTION__, __LINE__); fflush(stdout);
  if (writeInProgress) writeCond->wait();
  writeThread->join();
  printf("%s.%s.%d: done waiting for write.\n", __FILE__, __FUNCTION__, __LINE__); fflush(stdout);
  writeLock->unlock();
}

inline bool pollRead()
{
  bool ret;
  readLock->lock();
  ret = !readInProgress;
  readLock->unlock();
  return ret;
}

inline bool pollWrite()
{
  bool ret;
  writeLock->lock();
  ret = !writeInProgress;
  writeLock->unlock();
  return ret;
}


int main(int argc, char ** argv)
{
  if (argc != 4)
  {
    fprintf(stderr, "Usage: %s num_kernel_runs num_images_in_core total_images\n", *argv);
    return 1;
  }
  sscanf(argv[1], "%d", &NUM_PASSES_PER_IMAGE);
  sscanf(argv[2], "%d", &NUM_IMAGES_IN_CORE);
  sscanf(argv[3], "%d", &TOTAL_IMAGES);

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
  Image * cpuInImages0,  * cpuInImages1,  * gpuInImages0,  * gpuInImages1;
  Image * cpuOutImages0, * cpuOutImages1, * gpuOutImages0, * gpuOutImages1;
  unsigned char * imageRaw = new unsigned char[IMAGE_DIM * IMAGE_DIM];
  cudaStream_t kernelStream, memcpyStream;

  {
    unsigned char * pixels = new unsigned char[IMAGE_DIM * IMAGE_DIM];
    unsigned char * buf = pixels;
    FILE * fp = fopen("1024.pgm", "rb");
    fgets(reinterpret_cast<char * >(buf), 1023, fp);
    fgets(reinterpret_cast<char * >(buf), 1023, fp);
    fgets(reinterpret_cast<char * >(buf), 1023, fp);
    fread(pixels, IMAGE_DIM * IMAGE_DIM, 1, fp);
    imageTemplate = new Image;
    for (int i = 0; i < IMAGE_DIM; ++i)
    {
      for (int j = 0; j < IMAGE_DIM; ++j) imageTemplate->pixels[i][j] = static_cast<float>(*(buf++)) / 256.0f;
    }
    delete [] pixels;
  }

  readRunner  = new ReadImageRunner();
  writeRunner = new WriteImageRunner();
  readThread  = new Thread(readRunner);
  writeThread = new Thread(writeRunner);
  readLock    = new Lock();
  writeLock   = new Lock();
  readCond    = readLock->newCondition();
  writeCond   = writeLock->newCondition();

  CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_SAFE_CALL(cudaSetDevice(0));
  CUDA_SAFE_CALL(cudaStreamCreate(&kernelStream));
  CUDA_SAFE_CALL(cudaStreamCreate(&memcpyStream));

  CUDA_SAFE_CALL(cudaMallocHost           (reinterpret_cast<void ** >(&cpuKernel),      sizeof(kernel)            ));
  CUDA_SAFE_CALL(cudaMallocHost           (reinterpret_cast<void ** >(&cpuInImages0),   sizeof(Image) * NUM_IMAGES_IN_CORE));
  CUDA_SAFE_CALL(cudaMallocHost           (reinterpret_cast<void ** >(&cpuInImages1),   sizeof(Image) * NUM_IMAGES_IN_CORE));
  CUDA_SAFE_CALL(cudaMallocHost           (reinterpret_cast<void ** >(&cpuOutImages0),  sizeof(Image) * NUM_IMAGES_IN_CORE));
  CUDA_SAFE_CALL(cudaMallocHost           (reinterpret_cast<void ** >(&cpuOutImages1),  sizeof(Image) * NUM_IMAGES_IN_CORE));
  CUDA_SAFE_CALL(cudaMalloc               (reinterpret_cast<void ** >(&gpuKernel),      sizeof(kernel)            ));
  CUDA_SAFE_CALL(cudaMalloc               (reinterpret_cast<void ** >(&gpuInImages0),   sizeof(Image) * NUM_IMAGES_IN_CORE));
  CUDA_SAFE_CALL(cudaMalloc               (reinterpret_cast<void ** >(&gpuInImages1),   sizeof(Image) * NUM_IMAGES_IN_CORE));
  CUDA_SAFE_CALL(cudaMalloc               (reinterpret_cast<void ** >(&gpuOutImages0),  sizeof(Image) * NUM_IMAGES_IN_CORE));
  CUDA_SAFE_CALL(cudaMalloc               (reinterpret_cast<void ** >(&gpuOutImages1),  sizeof(Image) * NUM_IMAGES_IN_CORE));

  memcpy(cpuKernel, kernel, sizeof(kernel));

  CUDA_SAFE_CALL(cudaMemcpyAsync(gpuKernel, cpuKernel, sizeof(kernel), cudaMemcpyHostToDevice, memcpyStream));

  //  basic algorithm:
  //  read files synchronously
  //  start loop
  //    if we aren't on first iteration
  //      wait for kernel to finish
  //      wait for outputting images to finish
  //      save output from last kernel run
  //    schedule kernels
  //    if we aren't on last iteration
  //      asynchronously read more files
  //  end loop

  readImages(cpuInImages0, imageRaw);
  waitForRead();
  t0.start();

  for (int rep = 0; rep < (TOTAL_IMAGES + NUM_IMAGES_IN_CORE - 1) / NUM_IMAGES_IN_CORE; ++rep)
  {
    int numImagesInCore = NUM_IMAGES_IN_CORE;
    if (rep * NUM_IMAGES_IN_CORE > TOTAL_IMAGES) numImagesInCore = rep * NUM_IMAGES_IN_CORE - TOTAL_IMAGES;
    // printf("starting rep %d\n", rep); fflush(stdout);
    bool readDone         = false;
    bool copyDownStarted  = false;
    bool copyDownDone     = false;
    bool kernelDone       = rep < 1;
    bool copyUpStarted    = rep < 1;
    bool copyUpDone       = rep < 1;
    bool newWriteStarted  = rep < 1;
    bool oldWriteDone     = rep < 2;
/*
    const char * const descs[] = { "readDone", "copyDownStarted", "copyDownDone", "kernelDone", "copyUpStarted", "copyUpDone", "oldWriteDone", "newWriteStarted", "writeInProgress", "readInProgress" };
    bool lastFlags[] = { readDone, copyDownStarted, copyDownDone, kernelDone, copyUpStarted, copyUpDone, oldWriteDone, newWriteStarted, writeInProgress, readInProgress };*/
    while (!oldWriteDone || !readDone || !kernelDone || !copyDownDone || !copyUpDone || !newWriteStarted)
    {/*
      bool newFlags[] = { readDone, copyDownStarted, copyDownDone, kernelDone, copyUpStarted, copyUpDone, oldWriteDone, newWriteStarted, writeInProgress, readInProgress };
      bool changed = false;
      for (int i = 0; i < 8 && !changed; ++i) changed = (lastFlags[i] != newFlags[i]);
      if (changed)
      {
        for (int i = 0; i < 10; ++i) printf("  %15s: %s\n", descs[i], newFlags[i] ? "true" : "false");
        printf("\n");
        fflush(stdout);
      }
      for (int i = 0; i < 8; ++i) lastFlags[i] = newFlags[i];*/
      if (!kernelDone)
      {
        cudaError_t kernelStreamStatus = cudaStreamQuery(kernelStream);
        if (kernelStreamStatus != cudaErrorNotReady) // kernel is done executing.
        {
          printf("%.3f ms: kernel   - end.\n", t0.getElapsedMilliseconds());
          kernelDone = true;
          CUDA_SAFE_CALL(cudaMemcpyAsync(cpuOutImages0, gpuOutImages1, sizeof(Image) * NUM_IMAGES_IN_CORE, cudaMemcpyDeviceToHost, memcpyStream));
          printf("%.3f ms: copyUp   - start.\n", t0.getElapsedMilliseconds());
          copyUpStarted = true;
        }
      }
      if (!readDone)
      {
        readDone = pollRead();
        if (readDone)
        {
          printf("%.3f ms: read     - end.\n", t0.getElapsedMilliseconds());
          printf("%.3f ms: copyDown - start.\n", t0.getElapsedMilliseconds());
          CUDA_SAFE_CALL(cudaMemcpyAsync(gpuInImages0, cpuInImages0, sizeof(Image) * numImagesInCore, cudaMemcpyHostToDevice, memcpyStream));
          copyDownStarted = true;
        }
      }
      if (copyDownStarted && copyUpStarted && !copyDownDone && !copyUpDone)
      {
        cudaError_t memcpyStreamStatus = cudaStreamQuery(memcpyStream);
        if (memcpyStreamStatus != cudaErrorNotReady)
        {
          CUDA_SAFE_CALL(memcpyStreamStatus);
          printf("%.3f ms: copyUp   - end.\n", t0.getElapsedMilliseconds());
          printf("%.3f ms: copyDown - end.\n", t0.getElapsedMilliseconds());
          copyDownDone = copyUpDone = true;
          // printf("0 - calling write images.\n"); fflush(stdout);
          writeImages(cpuOutImages0, imageRaw, rep);
          printf("%.3f ms: write    - start.\n", t0.getElapsedMilliseconds());
        }
      }
      else if (copyDownStarted && !copyDownDone)
      {
        cudaError_t copyDownStreamStatus = cudaStreamQuery(memcpyStream);
        if (copyDownStreamStatus != cudaErrorNotReady)
        {
          CUDA_SAFE_CALL(copyDownStreamStatus);
          printf("%.3f ms: copyDown - end.\n", t0.getElapsedMilliseconds());
          copyDownDone = true;
        }
      }
      else if (copyUpStarted && !copyUpDone)
      {
        cudaError_t copyUpStreamStatus = cudaStreamQuery(memcpyStream);
        if (copyUpStreamStatus != cudaErrorNotReady)
        {
          CUDA_SAFE_CALL(copyUpStreamStatus);
          printf("%.3f ms: copyUp   - end.\n", t0.getElapsedMilliseconds());
          copyUpDone = true;
        }
      }
      if (!oldWriteDone && !newWriteStarted)
      {
        oldWriteDone = pollWrite();
        if (oldWriteDone) printf("%.3f ms: write    - end.\n", t0.getElapsedMilliseconds());
      }
      if (!newWriteStarted && oldWriteDone && copyUpDone)
      {
        // printf("1 - calling write images.\n"); fflush(stdout);
        printf("%.3f ms: write    - start.\n", t0.getElapsedMilliseconds());
        writeImages(cpuOutImages0, imageRaw, rep);
        newWriteStarted = true;
      }
    }
    // printf("scheduling kernels.\n"); fflush(stdout);
    cudaStreamSynchronize(memcpyStream);
    printf("%.3f ms: kernel   - start.\n", t0.getElapsedMilliseconds());
    scheduleKernels(gpuInImages0, gpuOutImages0, gpuKernel, kernelFactor, kernelStream, numImagesInCore);
    jswap(cpuInImages0,   cpuInImages1 );
    jswap(gpuInImages0,   gpuInImages1 );
    jswap(gpuOutImages0,  gpuOutImages1);
    if (rep != (TOTAL_IMAGES + NUM_IMAGES_IN_CORE - 1) / NUM_IMAGES_IN_CORE - 1) readImages(cpuInImages0, imageRaw);
  }
  CUDA_SAFE_CALL(cudaStreamSynchronize(kernelStream));
  printf("%.3f ms: kernel   - end.\n", t0.getElapsedMilliseconds());
  CUDA_SAFE_CALL(cudaStreamSynchronize(memcpyStream));

  writeLock->lock();
  if (writeInProgress) writeCond->wait();
  writeThread->join();
  writeLock->unlock();

  printf("%.3f ms: copyUp   - start.\n", t0.getElapsedMilliseconds());
  CUDA_SAFE_CALL(cudaMemcpy(cpuOutImages0, gpuOutImages1, sizeof(Image) * NUM_IMAGES_IN_CORE, cudaMemcpyDeviceToHost));
  printf("%.3f ms: copyUp   - end.\n", t0.getElapsedMilliseconds());
  printf("%.3f ms: write    - start.\n", t0.getElapsedMilliseconds());
  writeImages(cpuOutImages0, imageRaw, TOTAL_IMAGES);

  t0.stop();
  writeLock->lock();
  if (writeInProgress) writeCond->wait();
  printf("%.3f ms: write    - end.\n", t0.getElapsedMilliseconds());

  writeThread->join();
  writeLock->unlock();

  for (int i = 0; i < IMAGE_DIM; ++i)
  {
    for (int j = 0; j < IMAGE_DIM; ++j)
    {
      imageRaw[i * IMAGE_DIM + j] = clamp(cpuOutImages0[0].pixels[i][j] * 255.0f);
    }
  }

  fp = fopen("1024.out.pgm", "wb");
  fprintf(fp, "P5\n1024 1024\n255\n");
  fwrite(imageRaw, 1048576, 1, fp);
  fclose(fp);

  delete [] imageRaw;

  fprintf(stderr, "%4d %4d %4d %2d - %.3f ms \n", IMAGE_DIM, TOTAL_IMAGES, NUM_IMAGES_IN_CORE, NUM_PASSES_PER_IMAGE, t0.getElapsedMilliseconds());

  return 0;
}
