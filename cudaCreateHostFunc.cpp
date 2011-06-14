#include <cudaParams.h>
#include <vector>
#include <map>

static std::vector<std::vector<cudaParamType_t> > cudaHostParams;
static std::vector<cudaParamType_t>               cudaHostReturnTypes;
static std::vector<void * >                       cudaHostFuncPtrs;

const char * const CUDA_TYPE_STRINGS[] =
{
    "void",

    "dim3",
    "size_t",
    "char",                         "char2",            "char3",            "char4",
    "uchar",                        "uchar2",           "uchar3",           "uchar4",
    "short",                        "short2",           "short3",           "short4",
    "ushort",                       "ushort2",          "ushort3",          "ushort4",
    "int",                          "int2",             "int3",             "int4",
    "uint",                         "uint2",            "uint3",            "uint4",
    "long",                         "long2",            "long3",            "long4",
    "ulong",                        "ulong2",           "ulong3",           "ulong4",
    "float",                        "float2",           "float3",           "float4",
    "double",                       "double2",          "double3",          "double4",

    "dim3*",
    "size_t*",
    "char*",                     "char2*",        "char3*",        "char4*",
    "uchar*",                    "uchar2*",       "uchar3*",       "uchar4*",
    "short*",                    "short2*",       "short3*",       "short4*",
    "ushort*",                   "ushort2*",      "ushort3*",      "ushort4*",
    "int*",                      "int2*",         "int3*",         "int4*",
    "uint*",                     "uint2*",        "uint3*",        "uint4*",
    "long*",                     "long2*",        "long3*",        "long4*",
    "ulong*",                    "ulong2*",       "ulong3*",       "ulong4*",
    "float*",                    "float2*",       "float3*",       "float4*",
    "double*",                   "double2*",      "double3*",      "double4*",
};

const cudaParamType_t * cudaCallbackGetParams(const cudaHostFunction_t & func, size_t & numParams)
{
  if (func < 0 || func >= cudaHostParams.size())
  {
    numParams = ~(static_cast<size_t>(0));
    return NULL;
  }
  numParams = cudaHostParams[func].size();
  cudaParamType_t * ret = NULL;
  if (numParams > 0) ret = &cudaHostParams[func][0];
  return ret;
}
const cudaParamType_t * cudaCallbackGetReturnType(const cudaHostFunction_t & func)
{
  if (func < 0 || func >= cudaHostParams.size()) return NULL;
  return &cudaHostReturnTypes[func];
}
const void * cudaCallbackGetFuncPtr(const cudaHostFunction_t & func)
{
  if (func < 0 || func >= cudaHostParams.size()) return NULL;
  return cudaHostFuncPtrs[func];
}

cudaError_t cudaCreateHostFunc(cudaHostFunction_t * const func, void * funcPtr, const cudaParamType_t & returnParam, const int numParams, cudaParamType_t * params)
{
  if (func == NULL)                                                         return cudaInvalidHostFunctionPointer;
  if (funcPtr == NULL)                                                      return cudaInvalidFunctionPointer;
  if (numParams > 0 && params == NULL)                                      return cudaNullParameterList;
  if (returnParam < CUDA_PRIMITIVE_MIN || returnParam > CUDA_POINTER_MAX) return cudaErrorInvalidReturnType;
  for (int i = 0; i < numParams; ++i)
  {
    if (params[i] < CUDA_PRIMITIVE_MIN || params[i] >= CUDA_NUM_TYPES) return static_cast<cudaError_t>(cudaInvalidParameterType + i);
  }
  *func = static_cast<cudaHostFunction_t>(cudaHostParams.size());
  std::vector<cudaParamType_t> newParams;
  newParams.insert(newParams.begin(), params, params + numParams);
  cudaHostParams.push_back(newParams);
  cudaHostReturnTypes.push_back(returnParam);
  cudaHostFuncPtrs.push_back(funcPtr);

  return cudaSuccess;
}
