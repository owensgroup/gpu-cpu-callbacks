#include <cudaParam.h>
#include <cudaValue.h>
#include <cudaCallbackTypes.h>
#include <cstdio>
#include <typeinfo>

extern "C"
{
  __host__ void cudaCallbackCoerceArguments(char * storage, const cudaParamType_t * dstType, const cudaValue_t * srcValue)
  {
    CudaCallbackType * type = cudaCallbackCreateValue(srcValue);
    char * oldStorage = storage;
    switch (*dstType)
    {
    case CUDA_TYPE_CHAR:    *reinterpret_cast<char            * >(storage) = type->toChar();    break;
    case CUDA_TYPE_UCHAR:   *reinterpret_cast<unsigned char   * >(storage) = type->toUChar();   break;
    case CUDA_TYPE_SHORT:   *reinterpret_cast<short           * >(storage) = type->toShort();   break;
    case CUDA_TYPE_USHORT:  *reinterpret_cast<unsigned short  * >(storage) = type->toUShort();  break;
    case CUDA_TYPE_INT:     *reinterpret_cast<int             * >(storage) = type->toInt();     break;
    case CUDA_TYPE_LONG:    *reinterpret_cast<long            * >(storage) = type->toLong();    break;
    case CUDA_TYPE_SIZET:   *reinterpret_cast<size_t          * >(storage) = type->toULong();   break;
    case CUDA_TYPE_UINT:    *reinterpret_cast<unsigned int    * >(storage) = type->toUInt();    break;
    case CUDA_TYPE_ULONG:   *reinterpret_cast<unsigned long   * >(storage) = type->toULong();   break;
    case CUDA_TYPE_FLOAT:   *reinterpret_cast<float           * >(storage) = type->toFloat();   break;
    case CUDA_TYPE_DOUBLE:  *reinterpret_cast<double          * >(storage) = type->toDouble();  break;

    case CUDA_TYPE_CHAR2:   *reinterpret_cast<char2   * >(storage) = type->toChar2();   break;
    case CUDA_TYPE_UCHAR2:  *reinterpret_cast<uchar2  * >(storage) = type->toUChar2();  break;
    case CUDA_TYPE_SHORT2:  *reinterpret_cast<short2  * >(storage) = type->toShort2();  break;
    case CUDA_TYPE_USHORT2: *reinterpret_cast<ushort2 * >(storage) = type->toUShort2(); break;
    case CUDA_TYPE_INT2:    *reinterpret_cast<int2    * >(storage) = type->toInt2();    break;
    case CUDA_TYPE_UINT2:   *reinterpret_cast<uint2   * >(storage) = type->toUInt2();   break;
    case CUDA_TYPE_LONG2:   *reinterpret_cast<long2   * >(storage) = type->toLong2();   break;
    case CUDA_TYPE_ULONG2:  *reinterpret_cast<ulong2  * >(storage) = type->toULong2();  break;
    case CUDA_TYPE_FLOAT2:  *reinterpret_cast<float2  * >(storage) = type->toFloat2();  break;
    case CUDA_TYPE_DOUBLE2: *reinterpret_cast<double2 * >(storage) = type->toDouble2(); break;

    case CUDA_TYPE_DIM3:    *reinterpret_cast<dim3    * >(storage) = type->toDim3();    break;
    case CUDA_TYPE_CHAR3:   *reinterpret_cast<char3   * >(storage) = type->toChar3();   break;
    case CUDA_TYPE_UCHAR3:  *reinterpret_cast<uchar3  * >(storage) = type->toUChar3();  break;
    case CUDA_TYPE_SHORT3:  *reinterpret_cast<short3  * >(storage) = type->toShort3();  break;
    case CUDA_TYPE_USHORT3: *reinterpret_cast<ushort3 * >(storage) = type->toUShort3(); break;
    case CUDA_TYPE_INT3:    *reinterpret_cast<int3    * >(storage) = type->toInt3();    break;
    case CUDA_TYPE_UINT3:   *reinterpret_cast<uint3   * >(storage) = type->toUInt3();   break;
    case CUDA_TYPE_LONG3:   *reinterpret_cast<long3   * >(storage) = type->toLong3();   break;
    case CUDA_TYPE_ULONG3:  *reinterpret_cast<ulong3  * >(storage) = type->toULong3();  break;
    case CUDA_TYPE_FLOAT3:  *reinterpret_cast<float3  * >(storage) = type->toFloat3();  break;
    case CUDA_TYPE_DOUBLE3: *reinterpret_cast<double3 * >(storage) = type->toDouble3(); break;

    case CUDA_TYPE_CHAR4:   *reinterpret_cast<char4   * >(storage) = type->toChar4();   break;
    case CUDA_TYPE_UCHAR4:  *reinterpret_cast<uchar4  * >(storage) = type->toUChar4();  break;
    case CUDA_TYPE_SHORT4:  *reinterpret_cast<short4  * >(storage) = type->toShort4();  break;
    case CUDA_TYPE_USHORT4: *reinterpret_cast<ushort4 * >(storage) = type->toUShort4(); break;
    case CUDA_TYPE_INT4:    *reinterpret_cast<int4    * >(storage) = type->toInt4();    break;
    case CUDA_TYPE_UINT4:   *reinterpret_cast<uint4   * >(storage) = type->toUInt4();   break;
    case CUDA_TYPE_LONG4:   *reinterpret_cast<long4   * >(storage) = type->toLong4();   break;
    case CUDA_TYPE_ULONG4:  *reinterpret_cast<ulong4  * >(storage) = type->toULong4();  break;
    case CUDA_TYPE_FLOAT4:  *reinterpret_cast<float4  * >(storage) = type->toFloat4();  break;
    case CUDA_TYPE_DOUBLE4: *reinterpret_cast<double4 * >(storage) = type->toDouble4(); break;

    default:
      // if      (dstType->type >= CUDA_ARRAY_MIN    && dstType->type <= CUDA_ARRAY_MAX    ||
      //          dstType->type >= CUDA_POINTER_MIN  && dstType->type <= CUDA_POINTER_MAX  ||
      //          dstType->type == CUDA_TYPE_STRING)
      {
        *reinterpret_cast<void ** >(storage) = type->toPointer();
      }
    }
    // printf("%s.%s.%d: type(%s) src dst { %s(%p) %s(%p) }\n", __FILE__, __FUNCTION__, __LINE__, type->toString(), CUDA_TYPE_STRINGS[srcValue->type], *(char ** )storage, CUDA_TYPE_STRINGS[*dstType], *(char **)srcValue->mem); fflush(stdout);
#if 0
    {
      unsigned int * t = reinterpret_cast<unsigned int * >(oldStorage);
      fprintf(stderr, "type src dst storage { %s %s %s %p %p %p %p }\n",
                      type->toString(),
                      CUDA_TYPE_STRINGS[srcValue->type], CUDA_TYPE_STRINGS[dstType->type],
                      t[0], t[1], t[2], t[3]);
      fflush(stderr);
    }
#endif
    delete type;
  }
}
