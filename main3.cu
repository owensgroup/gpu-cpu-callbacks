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

extern "C"
{
  void __cdecl marshall(void * retPtr, void * params[])
  {
  }
}

__global__ void kernel(cudaCallbackData * cdata)
{
}

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
      fread(ret->data, height * width, 1, fp);
      fclose(fp);
      return ret;
    }
    void * row(const int rowIndex)
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

__shared__ unsigned int shImage[512];
__shared__ unsigned int storage[4];

__global__ void downsampleBy4(char * data, const int rows, const int cols)
{
  const int row = blockIdx.x * 4;
  const int col = (blockIdx.y * blockDim.x + threadIdx.x) * 4;

  unsigned int pix0 = *reinterpret_cast<unsigned int * >(data + (row + 0) * cols + col);
  unsigned int pix1 = *reinterpret_cast<unsigned int * >(data + (row + 1) * cols + col);
  unsigned int pix2 = *reinterpret_cast<unsigned int * >(data + (row + 2) * cols + col);
  unsigned int pix3 = *reinterpret_cast<unsigned int * >(data + (row + 3) * cols + col);

  pix0 = ((pix0 >> 24) & 0xFF) +
         ((pix0 >> 16) & 0xFF) +
         ((pix0 >>  8) & 0xFF) +
         ((pix0 >>  0) & 0xFF);
  pix1 = ((pix1 >> 24) & 0xFF) +
         ((pix1 >> 16) & 0xFF) +
         ((pix1 >>  8) & 0xFF) +
         ((pix1 >>  0) & 0xFF);
  pix2 = ((pix2 >> 24) & 0xFF) +
         ((pix2 >> 16) & 0xFF) +
         ((pix2 >>  8) & 0xFF) +
         ((pix2 >>  0) & 0xFF);
  pix3 = ((pix3 >> 24) & 0xFF) +
         ((pix3 >> 16) & 0xFF) +
         ((pix3 >>  8) & 0xFF) +
         ((pix3 >>  0) & 0xFF);

  pix0 = (pix0 + pix1 + pix2 + pix3) / 16;
  shImage[threadIdx.x] = pix0;
  __syncthreads();

  if (threadIdx.x < blockDim.x / 4)
  {
    unsigned int * output = reinterpret_cast<unsigned int * >(data + blockIdx.x * cols / 4 + blockIdx.y * blockDim.x + threadIdx.x * 4);
    *output = (shImage[threadIdx.x * 4 + 3] << 24)  |
              (shImage[threadIdx.x * 4 + 2] << 16)  |
              (shImage[threadIdx.x * 4 + 1] <<  8)  |
              (shImage[threadIdx.x * 4 + 0] <<  0) ;
  }

}

int main(int argc, char ** argv)
{
  const int MAX_MEM_TO_LOAD = 1024 * 1024 * 32;
  Image * image = Image::read(argv[1]);
  Image * output = new Image(image->rows / 4, image->cols / 4);
  char * gpuMem;
  printf("input  { %d %d }\n", image ->rows, image ->cols);
  printf("output { %d %d }\n", output->rows, output->cols);
  if (image->rows % 4 != 0)
  {
    printf("Error, input image rows must be divisible by four.\n");
    return 1;
  }
  if (image->cols % 2048 != 0)
  {
    printf("Error, input image cols must be divisible by 2048.\n");
    return 1;
  }
  if (image->cols * 4 > MAX_MEM_TO_LOAD)
  {
    printf("Error, input image cols can be at most %d.\n", MAX_MEM_TO_LOAD / 4);
    return 1;
  }

  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void ** >(&gpuMem), MAX_MEM_TO_LOAD));

  const char * inputPtr = reinterpret_cast<char * >(image->data);
  char * outputPtr = reinterpret_cast<char * >(output->data);
  int rowsLeft = image->rows;

  Timer t1;

  t1.start();
  while (rowsLeft > 0)
  {
    int currentRows = rowsLeft;
    if (rowsLeft * image->cols > MAX_MEM_TO_LOAD)
    {
      currentRows = MAX_MEM_TO_LOAD / image->cols;
    }

    dim3 gs(currentRows / 4, image->cols / 512 / 4, 1);
    dim3 bs(512, 1, 1);

    CUDA_SAFE_CALL(cudaMemcpy(gpuMem, inputPtr, currentRows * image->cols, cudaMemcpyHostToDevice));
    downsampleBy4<<<gs, bs>>>(gpuMem, currentRows, image->cols);
    CUDA_SAFE_CALL(cudaMemcpy(outputPtr, gpuMem, currentRows * image->cols / 16, cudaMemcpyDeviceToHost));

    inputPtr += currentRows * image->cols;
    outputPtr += currentRows * image->cols / 16;
    rowsLeft -= currentRows;
  }
  t1.stop();
  printf("image downsample took %.3f ms.\n", t1.getElapsedMilliseconds());

  CUDA_SAFE_CALL(cudaFree(gpuMem));

  output->write("gpu_out.pgm");

  delete image;
  delete output;

  return 0;
}
