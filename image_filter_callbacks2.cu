#include <callbacks.h>
#include <cstdio>
#include <cerrno>
#include <Timer.h>
#include <ThreadStuff.h>

#if 0
  const int IMAGE_DIM             = 1024;
  const int NUM_PASSES_PER_IMAGE  = 15;
  const int NUM_IMAGES_IN_CORE    = 30;
  const int TOTAL_IMAGES          = NUM_IMAGES_IN_CORE * 20;
#else
  const int IMAGE_DIM             = 1024;
  int NUM_PASSES_PER_IMAGE;
  int NUM_IMAGES_IN_CORE;
  int TOTAL_IMAGES;
  __device__ int gpuNumPasses;
  __device__ int gpuImagesInCore;
  __device__ int gpuTotalImages;
#endif

const int FILTER_BUFFER         = 2;
const int FILTER_ROWS           = FILTER_BUFFER * 2 + 1;
const int FILTER_COLS           = FILTER_BUFFER * 2 + 1;
const int INNER_DIM             = 16;
const int BLOCKS_PER_IMAGE      = (IMAGE_DIM / INNER_DIM) * (IMAGE_DIM / INNER_DIM);
const int NUM_SAMPLES           = (FILTER_BUFFER * 2 + INNER_DIM) * (FILTER_BUFFER * 2 + INNER_DIM);

typedef struct _Image
{
  float pixels[IMAGE_DIM][IMAGE_DIM];
} Image;

cudaStream_t memcpyStream;
int * imageStatus, * cpuImageStatus;
int * imageIDs, * gpuImageIDs;
Image * imageTemplate;
Image * cpuOutImages;
Image * gpuInImages, * gpuOutImages;

inline unsigned char clamp(const float f)
{
  return f < 0.0f ? 0 : f > 255.0f ? 255 : static_cast<unsigned char>(static_cast<unsigned int>(f));
}

void readInImage(const int imageIndex, const int imageID)
{
  cudaMemcpyAsync(gpuInImages + imageIndex, imageTemplate, sizeof(*imageTemplate), cudaMemcpyHostToDevice, memcpyStream);
  imageIDs[imageIndex] = imageID;
}

void writeOutImage(const int imageIndex, const int imageID)
{
  unsigned char * pixels = new unsigned char[IMAGE_DIM * IMAGE_DIM];
  unsigned char * buf = pixels;
  char filename[1024];
  sprintf(filename, "/tmp/out_%04d.pgm", imageIndex);
  FILE * fp = fopen(filename, "wb");
  fprintf(fp, "P5\n%d %d\n255\n", IMAGE_DIM, IMAGE_DIM);
  for (int i = 0; i < IMAGE_DIM; ++i)
  {
    for (int j = 0; j < IMAGE_DIM; ++j) *(buf++) = clamp(cpuOutImages[imageIndex].pixels[i][j] * 255.0f);
  }
  fwrite(pixels, IMAGE_DIM * IMAGE_DIM, 1, fp);
  fclose(fp);
  delete [] pixels;
}

void * writeImageThreadFunc(void * p)
{
  int * params = reinterpret_cast<int * >(p);
  const int imageIndex = params[0];
  const int imageID = params[1];
  writeOutImage(imageIndex, imageID);
  delete [] params;
  return NULL;
}

void finishedImage(float * params)
{
  static int inst = 0;
  if (inst++ % 1 == 0) printf("%.3f - finished image, saving { %d %d }\n", globalTimer.getElapsedSeconds(), static_cast<int>(params[0]), static_cast<int>(params[1]));
  const int imageIndex = static_cast<int>(params[0]);
  const int imageID    = static_cast<int>(params[1]);
  int * p = new int[2];
  p[0] = imageIndex;
  p[1] = imageID;
  pthread_t * thread = new pthread_t; // this is getting leaked, and im too lazy to make a global queue to delete these things.

  if (imageID + NUM_IMAGES_IN_CORE < TOTAL_IMAGES)
  {
    readInImage(imageIndex, imageID + NUM_IMAGES_IN_CORE);
  }
  cudaMemcpyAsync(cpuOutImages + imageIndex, gpuOutImages + imageIndex, sizeof(*imageTemplate), cudaMemcpyDeviceToHost, memcpyStream);
  cudaStreamSynchronize(memcpyStream);
  pthread_create(thread, NULL, writeImageThreadFunc, p);
}

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

#define DO_TIMING 0

__global__ void imageFilterKernel(Image * inImages, Image * outImages,
                                  int * imageStatus, int * imageIDs,
                                  const float * const kernel, const float kernelFactor,
                                  callbackData_t * cbData, callbackHandle_t imageFinishedCB)
{

  const int IMAGE_INDEX       = blockIdx.y % gpuImagesInCore;
  const int IMAGE_ID          = blockIdx.y;
  const int CHUNK_INDEX       = blockIdx.x % BLOCKS_PER_IMAGE;
  const int FILTER_PASS       = blockIdx.x / BLOCKS_PER_IMAGE;
  const int CHUNKS_PER_ROW    = (IMAGE_DIM / INNER_DIM);
  const int CHUNK_ROW_INDEX   = CHUNK_INDEX / CHUNKS_PER_ROW;
  const int CHUNK_COL_INDEX   = CHUNK_INDEX % CHUNKS_PER_ROW;
  const int ROW_INDEX         = CHUNK_ROW_INDEX * INNER_DIM;
  const int COL_INDEX         = CHUNK_COL_INDEX * INNER_DIM;
  volatile int * imgStatus    = imageStatus + IMAGE_INDEX;

  if (threadIdx.x == 0)
  {
    callbackDeviceData = cbData;
    volatile int * imageID      = const_cast<volatile int * >(imageIDs + IMAGE_INDEX);
    // hopefully we never wait, but just in case. this is the case where the CPU can't keep up with the demand of data
    // by the GPU and thus the image has not yet been read from disk.
#if !DO_TIMING
    while (*imageID != IMAGE_ID) { }
    // we really shouldn't ever wait here, but again just in case. this is the case where one block from the previous run has
    // yet to finish its execution. really odd case, considering that there should be a TON of other blocks working before
    // this block becomes active.
#endif
  }

  __syncthreads();

  if ((FILTER_PASS & 1) == 1) // we need to ping-pong between these guys.
  {
    Image * temp = inImages;
    inImages = outImages;
    outImages = temp;
  }

  // this is really lame that we have to do this. one would think that if we put something into shared mem, we could
  // keep it there for the next block. but nooooooo.
  for (int index = threadIdx.x; index < FILTER_ROWS * FILTER_COLS; index += blockDim.x)
  {
    KERNEL[index / FILTER_COLS][index % FILTER_COLS] = kernel[index];
  }

  // move the data from the input image into shared memory. yes, this is necessary because each pixel is accessed about
  // twenty-five times.
  moveInput(inImages + IMAGE_INDEX, ROW_INDEX, COL_INDEX);

  __syncthreads();

  // perform the convolution filter.
  const int X = threadIdx.x % INNER_DIM;
  const int Y = threadIdx.x / INNER_DIM;
  float sum = 0.0f;
  for (int row = -FILTER_BUFFER; row <= FILTER_BUFFER; ++row)
  {
    for (int col = -FILTER_BUFFER; col <= FILTER_BUFFER; ++col)
    {
      sum += samples[FILTER_BUFFER + row + Y][FILTER_BUFFER + col + X] * KERNEL[row + FILTER_BUFFER][col + FILTER_BUFFER];
    }
  }
  outImages[IMAGE_INDEX].pixels[ROW_INDEX + Y][COL_INDEX + X] = sum * kernelFactor;

  __syncthreads();
#if !DO_TIMING
  if (threadIdx.x == 0)
  {
    unsigned int oldVal;
    #if 1
      oldVal = atomicInc(reinterpret_cast<unsigned int * >(const_cast<int * >(imgStatus)), 0xFFFFFFFF);
    #else
      oldVal = blockIdx.x;
    #endif
    // this is the case where we are the last block to finish off the image. in that case, update ahoy.
    // printf("blockIdx.x oldVal bpm*nppm { %10d %10d %10d }\n", blockIdx.x, oldVal, BLOCKS_PER_IMAGE * NUM_PASSES_PER_IMAGE); fflush(stdout);
    if (oldVal % (BLOCKS_PER_IMAGE * gpuNumPasses) == BLOCKS_PER_IMAGE * gpuNumPasses - 1)
    {
      // printf("%d - executing callback.\n", blockIdx.x); fflush(stdout);
      callbackWait(IMAGE_INDEX);
      callbackExecuteAsync(true, IMAGE_INDEX, imageFinishedCB, IMAGE_INDEX, IMAGE_ID);
    }
  }
#endif
}

inline int getIndex(const int ind)
{
  return (ind + 3) % 3;
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

  globalTimer.start();
  Timer t0;
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
  callbackHostData_t * hostData;
  callbackHandle_t imageDoneHandle;
  cudaStream_t kernelStream;

  CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_SAFE_CALL(cudaSetDevice(0));

  CUDA_SAFE_CALL(cudaMemcpyToSymbol      (gpuNumPasses,     &NUM_PASSES_PER_IMAGE,  sizeof(NUM_PASSES_PER_IMAGE), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol      (gpuImagesInCore,  &NUM_IMAGES_IN_CORE,    sizeof(NUM_IMAGES_IN_CORE),   0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol      (gpuTotalImages,   &TOTAL_IMAGES,          sizeof(TOTAL_IMAGES),         0, cudaMemcpyHostToDevice));

  CUDA_SAFE_CALL(cudaHostAlloc           (reinterpret_cast<void ** >(&imageIDs),       sizeof(int)    * NUM_IMAGES_IN_CORE, cudaHostAllocMapped | cudaHostAllocPortable));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&gpuImageIDs),    imageIDs      , 0));

  CUDA_SAFE_CALL(cudaMallocHost          (reinterpret_cast<void ** >(&cpuOutImages),   sizeof(Image)  * NUM_IMAGES_IN_CORE));
  CUDA_SAFE_CALL(cudaMallocHost          (reinterpret_cast<void ** >(&cpuKernel),      sizeof(kernel)));
  CUDA_SAFE_CALL(cudaMallocHost          (reinterpret_cast<void ** >(&cpuImageStatus), sizeof(int)    * NUM_IMAGES_IN_CORE));
  CUDA_SAFE_CALL(cudaMallocHost          (reinterpret_cast<void ** >(&imageTemplate),  sizeof(*imageTemplate)));

  CUDA_SAFE_CALL(cudaMalloc              (reinterpret_cast<void ** >(&gpuInImages),    sizeof(Image)  * NUM_IMAGES_IN_CORE));
  CUDA_SAFE_CALL(cudaMalloc              (reinterpret_cast<void ** >(&gpuOutImages),   sizeof(Image)  * NUM_IMAGES_IN_CORE));
  CUDA_SAFE_CALL(cudaMalloc              (reinterpret_cast<void ** >(&gpuKernel),      sizeof(kernel)));
  CUDA_SAFE_CALL(cudaMalloc              (reinterpret_cast<void ** >(&imageStatus),    sizeof(int)    * NUM_IMAGES_IN_CORE));

  callbackHostInit(&hostData);
  imageDoneHandle = callbackRegister(hostData, finishedImage, 2);

  {
    unsigned char * pixels = new unsigned char[IMAGE_DIM * IMAGE_DIM];
    unsigned char * buf = pixels;
    FILE * fp = fopen("1024.pgm", "rb");
    fgets(reinterpret_cast<char * >(buf), 1023, fp);
    fgets(reinterpret_cast<char * >(buf), 1023, fp);
    fgets(reinterpret_cast<char * >(buf), 1023, fp);
    fread(pixels, IMAGE_DIM * IMAGE_DIM, 1, fp);
    for (int i = 0; i < IMAGE_DIM; ++i)
    {
      for (int j = 0; j < IMAGE_DIM; ++j) imageTemplate->pixels[i][j] = static_cast<float>(*(buf++)) / 256.0f;
    }
    delete [] pixels;
  }

  for (int i = 0; i < NUM_IMAGES_IN_CORE; ++i)
  {
    readInImage(i, i);
    cpuImageStatus[i] = 0;
  }

  memcpy(cpuKernel, kernel, sizeof(kernel));
  CUDA_SAFE_CALL(cudaMemcpy(gpuKernel, cpuKernel, sizeof(kernel), cudaMemcpyHostToDevice));

  t0.start();
  dim3 gs(BLOCKS_PER_IMAGE * NUM_PASSES_PER_IMAGE, TOTAL_IMAGES, 1);
  dim3 bs(INNER_DIM * INNER_DIM, 1, 1);
  printf("gs { %d %d %d }\n", gs.x, gs.y, gs.z);
  printf("bs { %d %d %d }\n", bs.x, bs.y, bs.z);
  fflush(stdout);
  CUDA_SAFE_CALL(cudaMemcpy(imageStatus, cpuImageStatus, sizeof(int) * NUM_IMAGES_IN_CORE, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaStreamCreate(&kernelStream));
  CUDA_SAFE_CALL(cudaStreamCreate(&memcpyStream));
  imageFilterKernel<<<gs, bs, 0, kernelStream>>>(gpuInImages, gpuOutImages,
                                                 imageStatus, gpuImageIDs,
                                                 gpuKernel, kernelFactor,
                                                 hostData->gpuCallbackData, imageDoneHandle);
  callbackSynchronize(hostData, kernelStream, -1);
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  t0.stop();
  CUDA_SAFE_CALL(cudaMemcpy(cpuImageStatus, imageStatus, sizeof(int) * NUM_IMAGES_IN_CORE, cudaMemcpyDeviceToHost));
  printf("image status\n");
  for (int i = 0; i < NUM_IMAGES_IN_CORE; i += 10)
  {
    for (int j = 0; j < 10 && i + j < NUM_IMAGES_IN_CORE; ++j)
    {
      printf("%10d  ", cpuImageStatus[i + j]);
    }
    printf("\n");
  }
  fflush(stdout);

  fprintf(stderr, "%4d %4d %4d %2d - %.3f ms \n", IMAGE_DIM, TOTAL_IMAGES, NUM_IMAGES_IN_CORE, NUM_PASSES_PER_IMAGE, t0.getElapsedMilliseconds());

  return 0;
}
