#include <cudaSafeCall.h>
#include <cudaParams.h>
#include <cudaParamCreate.h>
#include <cudaCallFunc.h>
#include <cstdio>
#include <cerrno>
#include <Timer.h>

#define USE_ZERO_COPY 0

const int IMAGE_DIM         = 1024;
const int FILTER_BUFFER     = 2;
const int FILTER_ROWS       = FILTER_BUFFER * 2 + 1;
const int FILTER_COLS       = FILTER_BUFFER * 2 + 1;
const int NUM_KERNEL_RUNS   = 10;
const int INNER_DIM         = 16;
const int NUM_IMAGES        = 60;
const int NUM_REPS          = 2;
const int BLOCKS_PER_IMAGE  = (IMAGE_DIM / INNER_DIM) * (IMAGE_DIM / INNER_DIM);
const int NUM_SAMPLES       = (FILTER_BUFFER * 2 + INNER_DIM) * (FILTER_BUFFER * 2 + INNER_DIM);
const int NUM_INNER_SAMPLES = INNER_DIM * INNER_DIM;

extern "C"
{
  void imageFinished(void * retPtr, void * params[])
  {
    int     fd  = *reinterpret_cast<int *     >(params[0]);
    void *  mem = *reinterpret_cast<void **   >(params[1]);
    size_t size = *reinterpret_cast<size_t *  >(params[2]);

    // ret = *reinterpret_cast<int * >(retPtr);
  }
}

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

int main(int argc, char ** argv)
{
  Timer t0;
  FILE * fp;
  char buf[1024];
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
  Image * cpuImages, * gpuImages;
  Image * cpuOutImages, * gpuOutImages;
  unsigned char * imageRaw = new unsigned char[1024 * 1024];

  CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_SAFE_CALL(cudaSetDevice(0));

#if USE_ZERO_COPY
  CUDA_SAFE_CALL(cudaHostAlloc            (reinterpret_cast<void ** >(&cpuKernel),    sizeof(kernel),             cudaHostAllocMapped));
  CUDA_SAFE_CALL(cudaHostAlloc            (reinterpret_cast<void ** >(&cpuImages),    sizeof(Image) * NUM_IMAGES, cudaHostAllocMapped));
  CUDA_SAFE_CALL(cudaHostAlloc            (reinterpret_cast<void ** >(&cpuOutImages), sizeof(Image) * NUM_IMAGES, cudaHostAllocMapped));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer (reinterpret_cast<void ** >(&gpuKernel),    cpuKernel,    0));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer (reinterpret_cast<void ** >(&gpuImages),    cpuImages,    0));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer (reinterpret_cast<void ** >(&gpuOutImages), cpuOutImages, 0));
#else
  CUDA_SAFE_CALL(cudaMallocHost           (reinterpret_cast<void ** >(&cpuKernel),    sizeof(kernel)            ));
  CUDA_SAFE_CALL(cudaMallocHost           (reinterpret_cast<void ** >(&cpuImages),    sizeof(Image) * NUM_IMAGES));
  CUDA_SAFE_CALL(cudaMallocHost           (reinterpret_cast<void ** >(&cpuOutImages), sizeof(Image) * NUM_IMAGES));
  CUDA_SAFE_CALL(cudaMalloc               (reinterpret_cast<void ** >(&gpuKernel),    sizeof(kernel)            ));
  CUDA_SAFE_CALL(cudaMalloc               (reinterpret_cast<void ** >(&gpuImages),    sizeof(Image) * NUM_IMAGES));
  CUDA_SAFE_CALL(cudaMalloc               (reinterpret_cast<void ** >(&gpuOutImages), sizeof(Image) * NUM_IMAGES));
#endif
  memcpy(cpuKernel, kernel, sizeof(kernel));

#if !USE_ZERO_COPY
    CUDA_SAFE_CALL(cudaMemcpy(gpuKernel, cpuKernel, sizeof(kernel),             cudaMemcpyHostToDevice));
#endif

  t0.start();

  for (int rep = 0; rep < NUM_REPS; ++rep)
  {
    for (int image = 0; image < NUM_IMAGES; ++image)
    {
      fp = fopen("1024.pgm", "rb");
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

#if !USE_ZERO_COPY
    CUDA_SAFE_CALL(cudaMemcpy(gpuKernel, cpuKernel, sizeof(kernel),             cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gpuImages, cpuImages, sizeof(Image) * NUM_IMAGES, cudaMemcpyHostToDevice));
#endif

    dim3 gridSize((IMAGE_DIM * IMAGE_DIM) / (INNER_DIM * INNER_DIM), 1, 1);
    dim3 blockSize(INNER_DIM * INNER_DIM, 1, 1);
    for (int i = 0; i < NUM_KERNEL_RUNS; ++i)
    {
      regularKernel<<<gridSize, blockSize>>>(NUM_IMAGES, gpuImages, gpuOutImages, gpuKernel, kernelFactor);
      Image * t = gpuImages;
      gpuImages = gpuOutImages;
      gpuOutImages = t;
    }
    CUDA_SAFE_CALL(cudaThreadSynchronize());

#if !USE_ZERO_COPY
    CUDA_SAFE_CALL(cudaMemcpy(cpuOutImages, gpuImages, sizeof(Image) * NUM_IMAGES, cudaMemcpyDeviceToHost));
#endif

    for (int image = 0; image < NUM_IMAGES; ++image)
    {
      char imgName[100];
      sprintf(imgName, "/tmp/image.%05d.%05d.pgm\n", rep, image);

      fp = fopen(imgName, "wb");
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
  }
  t0.stop();

  for (int i = 0; i < IMAGE_DIM; ++i)
  {
    for (int j = 0; j < IMAGE_DIM; ++j)
    {
      imageRaw[i * IMAGE_DIM + j] = clamp(cpuOutImages[0].pixels[i][j] * 255.0f);
    }
  }

  fp = fopen("1024.out.pgm", "wb");
  fprintf(fp, "P5\n1024 1024\n255\n");
  fwrite(imageRaw, 1048576, 1, fp);
  fclose(fp);

  delete [] imageRaw;

  printf("%d images took %.3f ms \n", NUM_IMAGES * NUM_REPS, t0.getElapsedMilliseconds());

  return 0;
}
