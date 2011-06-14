#ifndef __CUDAPARAMCREATE_H__
#define __CUDAPARAMCREATE_H__

#include <cudaParams.h>
#include <cstdio>

// a bunch of helper functions to easily create parameter types.

extern "C"
{
  inline cudaError_t cudaParamCreateVoid  (cudaParamType_t * param)
  {
    return cudaParamCreate(param, CUDA_TYPE_VOID);
  }

  inline bool cudaIsPrimitive (const cudaParamType_t type) { return type >= CUDA_PRIMITIVE_MIN  && type <= CUDA_PRIMITIVE_MAX;  }
  inline bool cudaIsPointer   (const cudaParamType_t type) { return type >= CUDA_POINTER_MIN    && type <= CUDA_POINTER_MAX;    }

}

extern "C++"
{

  template <typename T> inline cudaError_t cudaParamCreate(cudaParamType_t * param);
  template <typename T> inline cudaError_t cudaParamCreate(cudaParamType_t * param, const bool copyToHost, const bool copyToDevice);

  inline cudaError_t cudaParamCreate(cudaParamType_t * const param,
                                       const cudaParamType_t type)
  {
    *param = type;
    if (type < 0 || type >= CUDA_NUM_TYPES) return cudaErrorInvalidType;

    return cudaSuccess;
  }
  /////////////// Specialized functions go here. ///////////////

  // Primitives.

  // template <> inline cudaError_t cudaParamCreate<size_t>             (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_SIZET  ); }
  template <> inline cudaError_t cudaParamCreate<dim3>               (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_DIM3   ); }
  template <> inline cudaError_t cudaParamCreate<char>               (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_CHAR   ); }
  template <> inline cudaError_t cudaParamCreate<unsigned char>      (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_UCHAR  ); }
  template <> inline cudaError_t cudaParamCreate<short>              (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_SHORT  ); }
  template <> inline cudaError_t cudaParamCreate<unsigned short>     (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_USHORT ); }
  template <> inline cudaError_t cudaParamCreate<int>                (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_INT    ); }
  template <> inline cudaError_t cudaParamCreate<unsigned int>       (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_UINT   ); } // also is size_t
  template <> inline cudaError_t cudaParamCreate<long>               (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_LONG   ); }
  template <> inline cudaError_t cudaParamCreate<unsigned long>      (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_ULONG  ); }
  template <> inline cudaError_t cudaParamCreate<float>              (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_FLOAT  ); }
  template <> inline cudaError_t cudaParamCreate<double>             (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_DOUBLE ); }

  template <> inline cudaError_t cudaParamCreate<char2>              (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_CHAR2  ); }
  template <> inline cudaError_t cudaParamCreate<uchar2>             (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_UCHAR2 ); }
  template <> inline cudaError_t cudaParamCreate<short2>             (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_SHORT2 ); }
  template <> inline cudaError_t cudaParamCreate<ushort2>            (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_USHORT2); }
  template <> inline cudaError_t cudaParamCreate<int2>               (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_INT2   ); }
  template <> inline cudaError_t cudaParamCreate<uint2>              (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_UINT2  ); }
  template <> inline cudaError_t cudaParamCreate<long2>              (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_LONG2  ); }
  template <> inline cudaError_t cudaParamCreate<ulong2>             (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_ULONG2 ); }
  template <> inline cudaError_t cudaParamCreate<float2>             (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_FLOAT2 ); }
  template <> inline cudaError_t cudaParamCreate<double2>            (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_DOUBLE2); }

  template <> inline cudaError_t cudaParamCreate<char3>              (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_CHAR3  ); }
  template <> inline cudaError_t cudaParamCreate<uchar3>             (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_UCHAR3 ); }
  template <> inline cudaError_t cudaParamCreate<short3>             (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_SHORT3 ); }
  template <> inline cudaError_t cudaParamCreate<ushort3>            (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_USHORT3); }
  template <> inline cudaError_t cudaParamCreate<int3>               (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_INT3   ); }
  template <> inline cudaError_t cudaParamCreate<uint3>              (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_UINT3  ); }
  template <> inline cudaError_t cudaParamCreate<long3>              (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_LONG3  ); }
  template <> inline cudaError_t cudaParamCreate<ulong3>             (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_ULONG3 ); }
  template <> inline cudaError_t cudaParamCreate<float3>             (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_FLOAT3 ); }
  template <> inline cudaError_t cudaParamCreate<double3>            (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_DOUBLE3); }

  template <> inline cudaError_t cudaParamCreate<char4>              (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_CHAR4  ); }
  template <> inline cudaError_t cudaParamCreate<uchar4>             (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_UCHAR4 ); }
  template <> inline cudaError_t cudaParamCreate<short4>             (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_SHORT4 ); }
  template <> inline cudaError_t cudaParamCreate<ushort4>            (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_USHORT4); }
  template <> inline cudaError_t cudaParamCreate<int4>               (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_INT4   ); }
  template <> inline cudaError_t cudaParamCreate<uint4>              (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_UINT4  ); }
  template <> inline cudaError_t cudaParamCreate<long4>              (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_LONG4  ); }
  template <> inline cudaError_t cudaParamCreate<ulong4>             (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_ULONG4 ); }
  template <> inline cudaError_t cudaParamCreate<float4>             (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_FLOAT4 ); }
  template <> inline cudaError_t cudaParamCreate<double4>            (cudaParamType_t * param) { fflush(stdout); return cudaParamCreate(param, CUDA_TYPE_DOUBLE4); }

  // Pointers.

  // template <> inline cudaError_t cudaParamCreate<size_t * >          (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_SIZET_PTR  ); }
  template <> inline cudaError_t cudaParamCreate<dim3 * >            (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_DIM3_PTR   ); }
  template <> inline cudaError_t cudaParamCreate<char * >            (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_CHAR_PTR   ); }
  template <> inline cudaError_t cudaParamCreate<unsigned char * >   (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_UCHAR_PTR  ); }
  template <> inline cudaError_t cudaParamCreate<short * >           (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_SHORT_PTR  ); }
  template <> inline cudaError_t cudaParamCreate<unsigned short * >  (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_USHORT_PTR ); }
  template <> inline cudaError_t cudaParamCreate<int * >             (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_INT_PTR    ); }
  template <> inline cudaError_t cudaParamCreate<unsigned int * >    (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_UINT_PTR   ); }
  template <> inline cudaError_t cudaParamCreate<long * >            (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_LONG_PTR   ); }
  template <> inline cudaError_t cudaParamCreate<unsigned long * >   (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_ULONG_PTR  ); }
  template <> inline cudaError_t cudaParamCreate<float * >           (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_FLOAT_PTR  ); }
  template <> inline cudaError_t cudaParamCreate<double * >          (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_DOUBLE_PTR ); }

  template <> inline cudaError_t cudaParamCreate<char2 * >           (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_CHAR2_PTR  ); }
  template <> inline cudaError_t cudaParamCreate<uchar2 * >          (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_UCHAR2_PTR ); }
  template <> inline cudaError_t cudaParamCreate<short2 * >          (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_SHORT2_PTR ); }
  template <> inline cudaError_t cudaParamCreate<ushort2 * >         (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_USHORT2_PTR); }
  template <> inline cudaError_t cudaParamCreate<int2 * >            (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_INT2_PTR   ); }
  template <> inline cudaError_t cudaParamCreate<uint2 * >           (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_UINT2_PTR  ); }
  template <> inline cudaError_t cudaParamCreate<long2 * >           (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_LONG2_PTR  ); }
  template <> inline cudaError_t cudaParamCreate<ulong2 * >          (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_ULONG2_PTR ); }
  template <> inline cudaError_t cudaParamCreate<float2 * >          (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_FLOAT2_PTR ); }
  template <> inline cudaError_t cudaParamCreate<double2 * >         (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_DOUBLE2_PTR); }

  template <> inline cudaError_t cudaParamCreate<char3 * >           (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_CHAR3_PTR  ); }
  template <> inline cudaError_t cudaParamCreate<uchar3 * >          (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_UCHAR3_PTR ); }
  template <> inline cudaError_t cudaParamCreate<short3 * >          (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_SHORT3_PTR ); }
  template <> inline cudaError_t cudaParamCreate<ushort3 * >         (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_USHORT3_PTR); }
  template <> inline cudaError_t cudaParamCreate<int3 * >            (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_INT3_PTR   ); }
  template <> inline cudaError_t cudaParamCreate<uint3 * >           (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_UINT3_PTR  ); }
  template <> inline cudaError_t cudaParamCreate<long3 * >           (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_LONG3_PTR  ); }
  template <> inline cudaError_t cudaParamCreate<ulong3 * >          (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_ULONG3_PTR ); }
  template <> inline cudaError_t cudaParamCreate<float3 * >          (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_FLOAT3_PTR ); }
  template <> inline cudaError_t cudaParamCreate<double3 * >         (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_DOUBLE3_PTR); }

  template <> inline cudaError_t cudaParamCreate<char4 * >           (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_CHAR4_PTR  ); }
  template <> inline cudaError_t cudaParamCreate<uchar4 * >          (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_UCHAR4_PTR ); }
  template <> inline cudaError_t cudaParamCreate<short4 * >          (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_SHORT4_PTR ); }
  template <> inline cudaError_t cudaParamCreate<ushort4 * >         (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_USHORT4_PTR); }
  template <> inline cudaError_t cudaParamCreate<int4 * >            (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_INT4_PTR   ); }
  template <> inline cudaError_t cudaParamCreate<uint4 * >           (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_UINT4_PTR  ); }
  template <> inline cudaError_t cudaParamCreate<long4 * >           (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_LONG4_PTR  ); }
  template <> inline cudaError_t cudaParamCreate<ulong4 * >          (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_ULONG4_PTR ); }
  template <> inline cudaError_t cudaParamCreate<float4 * >          (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_FLOAT4_PTR ); }
  template <> inline cudaError_t cudaParamCreate<double4 * >         (cudaParamType_t * param) { return cudaParamCreate(param, CUDA_TYPE_DOUBLE4_PTR); }

}

#endif
