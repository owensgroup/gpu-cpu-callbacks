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

class Image
{
  public:
    int rows, cols;
    void * data;
    Image(const int numRows, const int numCols) : rows(numRows), cols(numCols)
    {
      data = new char[rows * cols];
    }
    ~Image()
    {
      delete [] reinterpret_cast<char * >(data);
    }
    static Image * read(const char * const fileName)
    {
      char p;
      int tag, width, height, colors;
      Image * ret;

      FILE * fp = fopen(fileName, "rb");
      fscanf(fp, "%c%d %d %d %d", &p, &tag, &width, &height, &colors);
      fgetc(fp);
      if (p != 'P' || tag != 5) return NULL;
      ret = new Image(height, width);
      char * ptr = reinterpret_cast<char * >(ret->data);
      int bytesRead = 0;
      while (bytesRead < height * width)
      {
        int toRead = height * width - bytesRead;
        if (toRead > 1024 * 1024 * 16) toRead = 1024 * 1024 * 16;
        printf("reading %d bytes from %s.\n", toRead, fileName); fflush(stdout);
        fread(ptr, toRead, 1, fp);
        bytesRead += toRead;
        ptr += toRead;
      }
      fclose(fp);
      printf("done reading %s.\n", fileName); fflush(stdout);
      return ret;
    }
    char * row(const int rowIndex)
    {
      return reinterpret_cast<char * >(data) + rowIndex * cols;
    }
    void write(const char * const fileName)
    {
      FILE * fp = fopen(fileName, "wb");
      fprintf(fp, "P5\n%d %d 255\n", rows, cols);
      fwrite(data, rows * cols, 1, fp);
      fclose(fp);
    }
};

int loadChunk(const int bidX, const int bidY, const int bdimX);
int writeChunk(const int chunk, const int bidX, const int bidY, const int bdimX);

extern "C"
{
  void __cdecl loadChunkMarshall(volatile void * retPtr, void * params[])
  {
    int bidX  = *reinterpret_cast<int * >(params[0]);
    int bidY  = *reinterpret_cast<int * >(params[1]);
    int bdimX = *reinterpret_cast<int * >(params[2]);
    *reinterpret_cast<volatile int * >(retPtr) = loadChunk(bidX, bidY, bdimX);
  }
  void __cdecl writeChunkMarshall(volatile void * retPtr, void * params[])
  {
    int     chunk = *reinterpret_cast<int * >(params[0]);
    int     bidX  = *reinterpret_cast<int * >(params[1]);
    int     bidY  = *reinterpret_cast<int * >(params[2]);
    int     bdimX = *reinterpret_cast<int * >(params[3]);
    *reinterpret_cast<volatile int * >(retPtr) = writeChunk(chunk, bidX, bidY, bdimX);
  }
}

Image * inputImage;
Image * outputImage;
pthread_mutex_t mutex;
const int MEM_PER_PAGE = 512 * 16;
const int NUM_PAGES = 512;
char * cpuMem, * gpuMem;
int gpuPages[NUM_PAGES];
volatile int nextFreePage = 0;

int loadChunk(const int bidX, const int bidY, const int bdimX)
{
  int ret = 0;
  int nfp = 0;
  pthread_mutex_lock(&mutex);
  // printf("%s.%s.%d: loading chunk into page %d\n", __FILE__, __FUNCTION__, __LINE__, nextFreePage); fflush(stderr);
  nfp = nextFreePage++;
  ret = gpuPages[nfp];
  pthread_mutex_unlock(&mutex);
  int row = bidX * 4;
  int col = bidY * bdimX;
  char * cpuPage = cpuMem + ret;
  // printf("bidx bidy cpu gpu { %d %d %p %p }\n", bidX, bidY, cpuPage, gpuMem + ret); fflush(stdout);
  memcpy(cpuPage,       inputImage->row(row + 0) + col, bdimX * 4);
  memcpy(cpuPage + col, inputImage->row(row + 1) + col, bdimX * 4);
  memcpy(cpuPage + col, inputImage->row(row + 2) + col, bdimX * 4);
  memcpy(cpuPage + col, inputImage->row(row + 3) + col, bdimX * 4);
  printf("loaded chunk for { %3d %3d } in page %d at offset %x\n", bidX, bidY, nfp, ret); fflush(stdout);
  return ret;
}
int writeChunk(const int chunkOffset, const int bidX, const int bidY, const int bdimX)
{
  int row = bidX;
  int col = bidY * bdimX / 4;
  char * cpuPage = cpuMem + chunkOffset;
  int nfp;
  // printf("bidx bidy cpu gpu { %d %d %p %p }\n", bidX, bidY, cpuPage, gpuMem + chunkOffset); fflush(stdout);
  memcpy(outputImage->row(row) + col, cpuPage, bdimX / 4);
  pthread_mutex_lock(&mutex);
  nfp = --nextFreePage;
  gpuPages[nfp] = chunkOffset;
  pthread_mutex_unlock(&mutex);
  printf("stored chunk for { %3d %3d } from page %d at offset %x\n", bidX, bidY, nfp, chunkOffset); fflush(stdout);
  return 0;
}

__shared__ unsigned int shImage[512];
__shared__ int offset;

__global__ void downsampleBy4(cudaCallbackData_t * cdata, cudaHostFunction_t loadChunkFunc, cudaHostFunction_t writeChunkFunc, const int rows, const int cols, char * mem)
{
  if (threadIdx.x == 0)
  {
    int bidX = blockIdx.x;
    int bidY = blockIdx.y;
    int bdimX = blockDim.x;
    offset = cudaCallFunc<int>(cdata, loadChunkFunc, bidX, bidY, bdimX);
  }
  __syncthreads();
  char * data = mem + offset;

#if 1
  unsigned int pix0 = *reinterpret_cast<unsigned int * >(data) + blockDim.x * 0 + threadIdx.x;
  unsigned int pix1 = *reinterpret_cast<unsigned int * >(data) + blockDim.x * 1 + threadIdx.x;
  unsigned int pix2 = *reinterpret_cast<unsigned int * >(data) + blockDim.x * 2 + threadIdx.x;
  unsigned int pix3 = *reinterpret_cast<unsigned int * >(data) + blockDim.x * 3 + threadIdx.x;

  pix0 = ((pix0 >> 24) & 0xFF) + ((pix0 >> 16) & 0xFF) + ((pix0 >> 8) & 0xFF) + (pix0 & 0xFF);
  pix1 = ((pix1 >> 24) & 0xFF) + ((pix1 >> 16) & 0xFF) + ((pix1 >> 8) & 0xFF) + (pix1 & 0xFF);
  pix2 = ((pix2 >> 24) & 0xFF) + ((pix2 >> 16) & 0xFF) + ((pix2 >> 8) & 0xFF) + (pix2 & 0xFF);
  pix3 = ((pix3 >> 24) & 0xFF) + ((pix3 >> 16) & 0xFF) + ((pix3 >> 8) & 0xFF) + (pix3 & 0xFF);

  pix0 = (pix0 + pix1 + pix2 + pix3) / 16;
  shImage[threadIdx.x] = pix0;

  __syncthreads();
  if (threadIdx.x < blockDim.x / 4)
  {
    unsigned int * output = reinterpret_cast<unsigned int * >(data) + threadIdx.x;
    *output = (shImage[threadIdx.x * 4 + 0] << 0) | (shImage[threadIdx.x * 4 + 1] << 8) | (shImage[threadIdx.x * 4 + 2] << 16) | (shImage[threadIdx.x * 4 + 3] << 24) ;
  }
#endif

  __syncthreads();
  if (threadIdx.x == 0)
  {
    int bidX = blockIdx.x;
    int bidY = blockIdx.y;
    int bdimX = blockDim.x;
    cudaCallFunc<int>(cdata, writeChunkFunc, offset, bidX, bidY, bdimX);
  }
}

int main(int argc, char ** argv)
{
  const int MAX_MEM_TO_LOAD = 1024 * 1024 * 32;
  inputImage = Image::read(argv[1]);
  if (inputImage->rows % 4 != 0)
  {
    printf("Error, input image rows must be divisible by four.\n");
    return 1;
  }
  if (inputImage->cols % 2048 != 0)
  {
    printf("Error, input image cols must be divisible by 2048.\n");
    return 1;
  }
  if (inputImage->cols * 4 > MAX_MEM_TO_LOAD)
  {
    printf("Error, input image cols can be at most %d.\n", MAX_MEM_TO_LOAD / 4);
    return 1;
  }
  outputImage = new Image(inputImage->rows / 4, inputImage->cols / 4);
  printf("input  { %d %d }\n", inputImage ->rows, inputImage ->cols);
  printf("output { %d %d }\n", outputImage->rows, outputImage->cols);
  fflush(stdout);
  cudaParamType_t params[4];
  cudaHostFunction_t loadFunc, storeFunc;
  cudaCallbackData_t * callbackData;
  cudaStream_t stream;

  pthread_mutex_init(&mutex, NULL);

  CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_SAFE_CALL(cudaSetDevice(1));
  CUDA_SAFE_CALL(cudaHostCallbackInit(&callbackData));

  CUDA_SAFE_CALL(cudaHostAlloc           (reinterpret_cast<void ** >(&cpuMem), MEM_PER_PAGE * NUM_PAGES,  cudaHostAllocMapped));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(reinterpret_cast<void ** >(&gpuMem), cpuMem,                    0));
  CUDA_SAFE_CALL(cudaStreamCreate        (&stream));

  for (int i = 0; i < NUM_PAGES; ++i) gpuPages[i] = MEM_PER_PAGE * i;

  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 0));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 1));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 2));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 3));
  CUDA_SAFE_CALL(cudaCreateHostFunc(&loadFunc, (void * )loadChunkMarshall, params[0], 3, params + 1));

  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 0));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 1));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 2));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 3));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 4));
  CUDA_SAFE_CALL(cudaCreateHostFunc(&storeFunc, (void * )writeChunkMarshall, params[0], 4, params + 1));

  printf("need to use grid size of (%d,%d) \n", inputImage->rows / 4, inputImage->cols / 512 / 4); fflush(stdout);
  // dim3 gs(32, 1, 1);
  dim3 gs(inputImage->rows / 8, inputImage->cols / 512 / 4, 1);
  dim3 bs(512, 1, 1);

  Timer t1;

  t1.start();

  downsampleBy4<<<gs, bs, 0, stream>>>(callbackData, loadFunc, storeFunc, inputImage->rows, inputImage->cols, gpuMem);
  CUDA_SAFE_CALL(cudaCallbackSynchronize(stream));
  t1.stop();
  printf("image downsample took %.3f ms.\n", t1.getElapsedMilliseconds()); fflush(stdout);
  outputImage->write("gpu_out.pgm");

  delete inputImage;
  delete outputImage;

  return 0;
}
