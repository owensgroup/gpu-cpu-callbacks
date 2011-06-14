#ifndef __CUDAPARAM_H__
#define __CUDAPARAM_H__

#include <cudaParamType.h>

#if 0 // all modern versions of CUDA have double3 and double4

extern "C"
{
  typedef struct double3
  {
    double x, y, z;
  } double3;
  typedef struct double4
  {
    double x, y, z, w;
  } double4;
}

#endif

#endif
