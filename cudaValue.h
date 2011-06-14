#ifndef __CUDAVALUE_H__
#define __CUDAVALUE_H__

#include <cudaHostFunction.h>
#include <cudaParamType.h>

extern "C"
{
  // A value coming up from the device via a callback.
  typedef struct cudaValue
  {
    // volatile char fence0[32];
    volatile cudaParamType_t type;         // What type of parameter was this? It may need to be converted.
    // volatile char fence1[32];
    volatile char mem[sizeof(double) * 4]; // Maximum size of an argument is that of the double4.
    // volatile char fence2[32];
  } cudaValue_t;
}

#endif
