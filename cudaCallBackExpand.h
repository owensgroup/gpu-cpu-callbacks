#ifndef __CUDACALLBACKEXPANDARGS0_H__
#define __CUDACALLBACKEXPANDARGS0_H__
/*
#define CUDA_CALL_BACK_EXPAND_BODY(
*/

// expands an argument list, i don't think i even use this anymore.
inline cudaError_t cudaCallBackExpandArgs0(cudaArgsList * argsList, void * returnSpace)
{
  if (cudaNumParams(argsList) == 0)
  {
    return cudaCallBackCallFunction0(returnSpace);
  }
  cudaParamType_t nextType = cudaCallBackParamType0(argsList);
  switch (nextType)
  {
  case CUDA_TYPE_VOID:    return cudaInvalidConfiguration;
  case CUDA_TYPE_CHAR:    case CUDA_TYPE_UCHAR:       return cudaCallBackExpandArgs1<char   >(argsList, returnSpace, *reinterpret_cast<char    * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_SHORT:   case CUDA_TYPE_USHORT:      return cudaCallBackExpandArgs1<short  >(argsList, returnSpace, *reinterpret_cast<short   * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_SIZET:
  case CUDA_TYPE_INT:     case CUDA_TYPE_UINT:
  case CUDA_TYPE_LONG:    case CUDA_TYPE_ULONG:       return cudaCallBackExpandArgs1<int    >(argsList, returnSpace, *reinterpret_cast<int     * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_FLOAT:                               return cudaCallBackExpandArgs1<float  >(argsList, returnSpace, *reinterpret_cast<float   * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_DOUBLE:                              return cudaCallBackExpandArgs1<double >(argsList, returnSpace, *reinterpret_cast<double  * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_CHAR2:   case CUDA_TYPE_UCHAR2:      return cudaCallBackExpandArgs1<char2  >(argsList, returnSpace, *reinterpret_cast<char2   * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_CHAR3:   case CUDA_TYPE_UCHAR3:      return cudaCallBackExpandArgs1<char3  >(argsList, returnSpace, *reinterpret_cast<char3   * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_CHAR4:   case CUDA_TYPE_UCHAR4:      return cudaCallBackExpandArgs1<char4  >(argsList, returnSpace, *reinterpret_cast<char4   * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_SHORT2:  case CUDA_TYPE_USHORT2:     return cudaCallBackExpandArgs1<short2 >(argsList, returnSpace, *reinterpret_cast<short2  * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_SHORT3:  case CUDA_TYPE_USHORT3:     return cudaCallBackExpandArgs1<short3 >(argsList, returnSpace, *reinterpret_cast<short3  * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_SHORT4:  case CUDA_TYPE_USHORT4:     return cudaCallBackExpandArgs1<short4 >(argsList, returnSpace, *reinterpret_cast<short4  * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_LONG2:   case CUDA_TYPE_ULONG2:
  case CUDA_TYPE_INT2:    case CUDA_TYPE_UINT2:       return cudaCallBackExpandArgs1<int2   >(argsList, returnSpace, *reinterpret_cast<int2    * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_LONG3:   case CUDA_TYPE_ULONG3:
  case CUDA_TYPE_INT3:    case CUDA_TYPE_UINT3:       return cudaCallBackExpandArgs1<int3   >(argsList, returnSpace, *reinterpret_cast<int3    * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_LONG4:   case CUDA_TYPE_ULONG4:
  case CUDA_TYPE_INT4:    case CUDA_TYPE_UINT4:       return cudaCallBackExpandArgs1<int4   >(argsList, returnSpace, *reinterpret_cast<int4    * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_DIM3:                                return cudaCallBackExpandArgs1<dim3   >(argsList, returnSpace, *reinterpret_cast<dim3    * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_FLOAT2:                              return cudaCallBackExpandArgs1<float2 >(argsList, returnSpace, *reinterpret_cast<float2  * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_FLOAT3:                              return cudaCallBackExpandArgs1<float3 >(argsList, returnSpace, *reinterpret_cast<float3  * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_FLOAT4:                              return cudaCallBackExpandArgs1<float4 >(argsList, returnSpace, *reinterpret_cast<float4  * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_DOUBLE2:                             return cudaCallBackExpandArgs1<double2>(argsList, returnSpace, *reinterpret_cast<double2 * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_DOUBLE3:                             return cudaCallBackExpandArgs1<double3>(argsList, returnSpace, *reinterpret_cast<double3 * >(cudaCallBackParamMem0(argsList)));
  case CUDA_TYPE_DOUBLE4:                             return cudaCallBackExpandArgs1<double4>(argsList, returnSpace, *reinterpret_cast<double4 * >(cudaCallBackParamMem0(argsList)));
  default:
    if      (nextType >= CUDA_ARRAY_MIN && nextType <= CUDA_ARRAY_MAX)
    {
      return cudaInvalidConfiguration;
    }
    else if (nextType >= CUDA_POINTER_MIN && nextType <= CUDA_POINTER_MAX)
    {
      return cudaInvalidConfiguration;
    }
    else if (nextType == CUDA_TYPE_STRING)
    {
      return cudaInvalidConfiguration;
    }
    return cudaInvalidConfiguration;
  }
}
/*
template <typename T1>
inline cudaError_t cudaCallBackExpandArgs1(cudaCallBackExpandArgs2, T1 t1, cudaArgsList * argsList, void * returnSpace)
{
  CUDA_CALL_BACK_EXPAND_BODY(1, argsList, returnSpace, t1);
}

template <typename T1, typename T2>
inline cudaError_t cudaCallBackExpandArgs2(cudaCallBackExpandArgs3, T1 t1, cudaArgsList * argsList, void * returnSpace)
{
  CUDA_CALL_BACK_EXPAND_BODY(2, argsList, returnSpace, t1);
}
*/
#endif
