#ifndef __CUDAPARAMTYPE_H__
#define __CUDAPARAMTYPE_H__

// All the different types out there that might be passed in via callbacks. We
// also accept vector types in CUDA.

extern const char * const CUDA_TYPE_STRINGS[];

extern "C++"
{
  typedef enum cudaParamType
  {
    CUDA_TYPE_VOID = 0,

    CUDA_PRIMITIVE_MIN,
    CUDA_TYPE_DIM3 = CUDA_PRIMITIVE_MIN,
    CUDA_TYPE_SIZET,
    CUDA_TYPE_CHAR,                         CUDA_TYPE_CHAR2,            CUDA_TYPE_CHAR3,            CUDA_TYPE_CHAR4,
    CUDA_TYPE_UCHAR,                        CUDA_TYPE_UCHAR2,           CUDA_TYPE_UCHAR3,           CUDA_TYPE_UCHAR4,
    CUDA_TYPE_SHORT,                        CUDA_TYPE_SHORT2,           CUDA_TYPE_SHORT3,           CUDA_TYPE_SHORT4,
    CUDA_TYPE_USHORT,                       CUDA_TYPE_USHORT2,          CUDA_TYPE_USHORT3,          CUDA_TYPE_USHORT4,
    CUDA_TYPE_INT,                          CUDA_TYPE_INT2,             CUDA_TYPE_INT3,             CUDA_TYPE_INT4,
    CUDA_TYPE_UINT,                         CUDA_TYPE_UINT2,            CUDA_TYPE_UINT3,            CUDA_TYPE_UINT4,
    CUDA_TYPE_LONG,                         CUDA_TYPE_LONG2,            CUDA_TYPE_LONG3,            CUDA_TYPE_LONG4,
    CUDA_TYPE_ULONG,                        CUDA_TYPE_ULONG2,           CUDA_TYPE_ULONG3,           CUDA_TYPE_ULONG4,
    CUDA_TYPE_FLOAT,                        CUDA_TYPE_FLOAT2,           CUDA_TYPE_FLOAT3,           CUDA_TYPE_FLOAT4,
    CUDA_TYPE_DOUBLE,                       CUDA_TYPE_DOUBLE2,          CUDA_TYPE_DOUBLE3,          CUDA_TYPE_DOUBLE4,
    CUDA_PRIMITIVE_MAX = CUDA_TYPE_DOUBLE4,

    CUDA_POINTER_MIN,
    CUDA_TYPE_DIM3_PTR = CUDA_POINTER_MIN,
    CUDA_TYPE_SIZET_PTR,
    CUDA_TYPE_CHAR_PTR,                     CUDA_TYPE_CHAR2_PTR,        CUDA_TYPE_CHAR3_PTR,        CUDA_TYPE_CHAR4_PTR,
    CUDA_TYPE_UCHAR_PTR,                    CUDA_TYPE_UCHAR2_PTR,       CUDA_TYPE_UCHAR3_PTR,       CUDA_TYPE_UCHAR4_PTR,
    CUDA_TYPE_SHORT_PTR,                    CUDA_TYPE_SHORT2_PTR,       CUDA_TYPE_SHORT3_PTR,       CUDA_TYPE_SHORT4_PTR,
    CUDA_TYPE_USHORT_PTR,                   CUDA_TYPE_USHORT2_PTR,      CUDA_TYPE_USHORT3_PTR,      CUDA_TYPE_USHORT4_PTR,
    CUDA_TYPE_INT_PTR,                      CUDA_TYPE_INT2_PTR,         CUDA_TYPE_INT3_PTR,         CUDA_TYPE_INT4_PTR,
    CUDA_TYPE_UINT_PTR,                     CUDA_TYPE_UINT2_PTR,        CUDA_TYPE_UINT3_PTR,        CUDA_TYPE_UINT4_PTR,
    CUDA_TYPE_LONG_PTR,                     CUDA_TYPE_LONG2_PTR,        CUDA_TYPE_LONG3_PTR,        CUDA_TYPE_LONG4_PTR,
    CUDA_TYPE_ULONG_PTR,                    CUDA_TYPE_ULONG2_PTR,       CUDA_TYPE_ULONG3_PTR,       CUDA_TYPE_ULONG4_PTR,
    CUDA_TYPE_FLOAT_PTR,                    CUDA_TYPE_FLOAT2_PTR,       CUDA_TYPE_FLOAT3_PTR,       CUDA_TYPE_FLOAT4_PTR,
    CUDA_TYPE_DOUBLE_PTR,                   CUDA_TYPE_DOUBLE2_PTR,      CUDA_TYPE_DOUBLE3_PTR,      CUDA_TYPE_DOUBLE4_PTR,
    CUDA_POINTER_MAX = CUDA_TYPE_DOUBLE4_PTR,

    CUDA_NUM_TYPES,
  } cudaParamType_t;
}

#endif
