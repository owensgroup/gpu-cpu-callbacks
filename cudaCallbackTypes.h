#ifndef __CUDACALLBACK_TYPES_H__
#define __CUDACALLBACK_TYPES_H__

#include <cstddef>
#include <cstdio>
#include <vector_types.h>

#ifdef __LP64__ // needed for 64 bit linux. something to do with the size of longs most likely.

#if 0 // all modern versions of CUDA have these now.
struct  long4 {          long int x, y, z, w; };
struct ulong4 { unsigned long int x, y, z, w; };
struct  long3 {          long int x, y, z; }   ;
struct ulong3 { unsigned long int x, y, z; }   ;
#endif

#endif

// Some C++ conversion stuff, just ignore this class, you don't need it.

typedef int callbackAsyncRequest_t;

class CudaCallbackType
{
  protected:
    mutable char canon[200];
    cudaParamType_t type;
    CudaCallbackType(cudaParamType_t pType) : type(pType) { }
  public:
    virtual ~CudaCallbackType() { }

    // virtual void assign(const CudaCallbackType * const rhs) = 0;

    cudaParamType_t getType() const { return type; }
    virtual char           toChar()    const { return toLong()  & 0xFF; }
    virtual unsigned char  toUChar()   const { return toULong() & 0xFF; }
    virtual short          toShort()   const { return toLong()  & 0xFFFF; }
    virtual unsigned short toUShort()  const { return toULong() & 0xFFFF; }
    virtual int            toInt()     const { return toLong()  & 0xFFFFFFFF; }
    virtual unsigned int   toUInt()    const { return toULong() & 0xFFFFFFFF; }
    virtual long           toLong()    const = 0;
    virtual unsigned long  toULong()   const = 0;
    virtual size_t         toSizeT()   const { return toULong(); }
    virtual float          toFloat()   const { return (float)toDouble(); }
    virtual double         toDouble()  const = 0;
    virtual void *         toPointer() const = 0;

    virtual char1          toChar1()   const { char1   ret = { toLong4()   .x & 0x000000FF }; return ret; }
    virtual uchar1         toUChar1()  const { uchar1  ret = { toULong4()  .x & 0x000000FF }; return ret; }
    virtual short1         toShort1()  const { short1  ret = { toLong4()   .x & 0x0000FFFF }; return ret; }
    virtual ushort1        toUShort1() const { ushort1 ret = { toULong4()  .x & 0x0000FFFF }; return ret; }
    virtual int1           toInt1()    const { int1    ret = { toLong4()   .x & 0xFFFFFFFF }; return ret; }
    virtual uint1          toUInt1()   const { uint1   ret = { toULong4()  .x & 0xFFFFFFFF }; return ret; }
    virtual long1          toLong1()   const { long1   ret = { toLong4()   .x              }; return ret; }
    virtual ulong1         toULong1()  const { ulong1  ret = { toULong4()  .x              }; return ret; }
    virtual float1         toFloat1()  const { float1  ret = { toFloat4()  .x              }; return ret; }
    virtual double1        toDouble1() const { double1 ret = { toDouble4() .x              }; return ret; }

    virtual char2          toChar2()   const {  long4  t = toLong4();    char2   ret = { t.x & 0x000000FF, t.y & 0x000000FF }; return ret; }
    virtual uchar2         toUChar2()  const { ulong4  t = toULong4();   uchar2  ret = { t.x & 0x000000FF, t.y & 0x000000FF }; return ret; }
    virtual short2         toShort2()  const {  long4  t = toLong4();    short2  ret = { t.x & 0x0000FFFF, t.y & 0x0000FFFF }; return ret; }
    virtual ushort2        toUShort2() const { ulong4  t = toULong4();   ushort2 ret = { t.x & 0x0000FFFF, t.y & 0x0000FFFF }; return ret; }
    virtual int2           toInt2()    const {  long4  t = toLong4();    int2    ret = { t.x & 0xFFFFFFFF, t.y & 0xFFFFFFFF }; return ret; }
    virtual uint2          toUInt2()   const { ulong4  t = toULong4();   uint2   ret = { t.x & 0xFFFFFFFF, t.y & 0xFFFFFFFF }; return ret; }
    virtual long2          toLong2()   const {  long4  t = toLong4();    long2   ret = { t.x             , t.y              }; return ret; }
    virtual ulong2         toULong2()  const { ulong4  t = toULong4();   ulong2  ret = { t.x             , t.y              }; return ret; }
    virtual float2         toFloat2()  const { float4  t = toFloat4();   float2  ret = { t.x             , t.y              }; return ret; }
    virtual double2        toDouble2() const { double4 t = toDouble4();  double2 ret = { t.x             , t.y              }; return ret; }

    virtual char3          toChar3()   const {  long4  t = toLong4();    char3   ret = { t.x & 0x000000FF, t.y & 0x000000FF, t.z & 0x000000FF }; return ret; }
    virtual uchar3         toUChar3()  const { ulong4  t = toULong4();   uchar3  ret = { t.x & 0x000000FF, t.y & 0x000000FF, t.z & 0x000000FF }; return ret; }
    virtual short3         toShort3()  const {  long4  t = toLong4();    short3  ret = { t.x & 0x0000FFFF, t.y & 0x0000FFFF, t.z & 0x0000FFFF }; return ret; }
    virtual ushort3        toUShort3() const { ulong4  t = toULong4();   ushort3 ret = { t.x & 0x0000FFFF, t.y & 0x0000FFFF, t.z & 0x0000FFFF }; return ret; }
    virtual int3           toInt3()    const {  long4  t = toLong4();    int3    ret = { t.x & 0xFFFFFFFF, t.y & 0xFFFFFFFF, t.z & 0xFFFFFFFF }; return ret; }
    virtual uint3          toUInt3()   const { ulong4  t = toULong4();   uint3   ret = { t.x & 0xFFFFFFFF, t.y & 0xFFFFFFFF, t.z & 0xFFFFFFFF }; return ret; }
    virtual dim3           toDim3()    const { ulong4  t = toULong4();   dim3    ret   ( t.x             , t.y             , t.z              ); return ret; }
    virtual long3          toLong3()   const {  long4  t = toLong4();    long3   ret = { t.x             , t.y             , t.z              }; return ret; }
    virtual ulong3         toULong3()  const { ulong4  t = toULong4();   ulong3  ret = { t.x             , t.y             , t.z              }; return ret; }
    virtual float3         toFloat3()  const { float4  t = toFloat4();   float3  ret = { t.x             , t.y             , t.z              }; return ret; }
    virtual double3        toDouble3() const { double4 t = toDouble4();  double3 ret = { t.x             , t.y             , t.z              }; return ret; }

    virtual char4          toChar4()   const {  long4  t = toLong4();    char4   ret = { t.x & 0x000000FF, t.y & 0x000000FF, t.z & 0x000000FF }; return ret; }
    virtual uchar4         toUChar4()  const { ulong4  t = toULong4();   uchar4  ret = { t.x & 0x000000FF, t.y & 0x000000FF, t.z & 0x000000FF }; return ret; }
    virtual short4         toShort4()  const {  long4  t = toLong4();    short4  ret = { t.x & 0x0000FFFF, t.y & 0x0000FFFF, t.z & 0x0000FFFF }; return ret; }
    virtual ushort4        toUShort4() const { ulong4  t = toULong4();   ushort4 ret = { t.x & 0x0000FFFF, t.y & 0x0000FFFF, t.z & 0x0000FFFF }; return ret; }
    virtual int4           toInt4()    const {  long4  t = toLong4();    int4    ret = { t.x & 0xFFFFFFFF, t.y & 0xFFFFFFFF, t.z & 0xFFFFFFFF }; return ret; }
    virtual uint4          toUInt4()   const { ulong4  t = toULong4();   uint4   ret = { t.x & 0xFFFFFFFF, t.y & 0xFFFFFFFF, t.z & 0xFFFFFFFF }; return ret; }
    virtual long4          toLong4()   const = 0;
    virtual ulong4         toULong4()  const = 0;
    virtual float4         toFloat4()  const { double4 t = toDouble4();  float4  ret = { static_cast<float>(t.x), static_cast<float>(t.y), static_cast<float>(t.z), static_cast<float>(t.w) }; return ret; }
    virtual double4        toDouble4() const = 0;

    virtual char * toString() const = 0;

};

class CudaChar : public CudaCallbackType
{
  public:
    char val;
    CudaChar(const char v = 0) : CudaCallbackType(CUDA_TYPE_CHAR), val(v) { }
    virtual char           toChar()    const { return val;                             }
    virtual unsigned char  toUChar()   const { return static_cast<unsigned char>(val); }
    virtual long           toLong()    const { return static_cast<long>(val);          }
    virtual unsigned long  toULong()   const { return static_cast<unsigned long>(val); }
    virtual double         toDouble()  const { return static_cast<double>(val);        }
    virtual void *         toPointer() const { return NULL; }
    virtual long4          toLong4()   const {  long4  ret = { static_cast<long         >(val), 0,   0,   0   }; return ret; }
    virtual ulong4         toULong4()  const { ulong4  ret = { static_cast<unsigned long>(val), 0,   0,   0   }; return ret; }
    virtual double4        toDouble4() const { double4 ret = { static_cast<double       >(val), 0.0, 0.0, 0.0 }; return ret; }

    virtual char * toString() const
    {
      sprintf(canon, "char(%c(%02Xh))", val <= ' ' ? '.' : val, val);
      return canon;
    }
};
class CudaUChar : public CudaCallbackType
{
  public:
    unsigned char val;
    CudaUChar(const unsigned char v = 0) : CudaCallbackType(CUDA_TYPE_UCHAR), val(v) { }
    virtual char           toChar()    const { return static_cast<char>(val);          }
    virtual unsigned char  toUChar()   const { return val;                             }
    virtual long           toLong()    const { return static_cast<long>(val);          }
    virtual unsigned long  toULong()   const { return static_cast<unsigned long>(val); }
    virtual double         toDouble()  const { return static_cast<double>(val);        }
    virtual void *         toPointer() const { return NULL; }
    virtual long4          toLong4()   const {  long4  ret = { static_cast<long         >(val), 0,   0,   0   }; return ret; }
    virtual ulong4         toULong4()  const { ulong4  ret = { static_cast<unsigned long>(val), 0,   0,   0   }; return ret; }
    virtual double4        toDouble4() const { double4 ret = { static_cast<double       >(val), 0.0, 0.0, 0.0 }; return ret; }
    virtual char * toString() const
    {
      sprintf(canon, "uchar(%c(%02Xh))", (val <= ' ' || val > 0x7F) ? '.' : val, val);
      return canon;
    }
};
class CudaShort : public CudaCallbackType
{
  public:
    short val;
    CudaShort(const short v = 0) : CudaCallbackType(CUDA_TYPE_SHORT), val(v) { }
    virtual short          toShort()   const { return val;                               }
    virtual unsigned short toUShort()  const { return static_cast<unsigned short>(val);  }
    virtual long           toLong()    const { return static_cast<long>(val);          }
    virtual unsigned long  toULong()   const { return static_cast<unsigned long>(val); }
    virtual double         toDouble()  const { return static_cast<double>(val);          }
    virtual void *         toPointer() const { return NULL;                              }
    virtual long4          toLong4()   const {  long4  ret = { static_cast<long         >(val), 0,   0,   0   }; return ret; }
    virtual ulong4         toULong4()  const { ulong4  ret = { static_cast<unsigned long>(val), 0,   0,   0   }; return ret; }
    virtual double4        toDouble4() const { double4 ret = { static_cast<double       >(val), 0.0, 0.0, 0.0 }; return ret; }
    virtual char * toString() const
    {
      sprintf(canon, "short(%d)", val);
      return canon;
    }
};
class CudaUShort : public CudaCallbackType
{
  public:
    unsigned short val;
    CudaUShort(const unsigned short v = 0) : CudaCallbackType(CUDA_TYPE_USHORT), val(v) { }
    virtual short          toShort()   const { return static_cast<short>(val);         }
    virtual unsigned short toUShort()  const { return val;                             }
    virtual long           toLong()    const { return static_cast<long>(val);          }
    virtual unsigned long  toULong()   const { return static_cast<unsigned long>(val); }
    virtual double         toDouble()  const { return static_cast<double>(val);        }
    virtual void *         toPointer() const { return NULL; }
    virtual long4          toLong4()   const {  long4  ret = { static_cast<long         >(val), 0,   0,   0   }; return ret; }
    virtual ulong4         toULong4()  const { ulong4  ret = { static_cast<unsigned long>(val), 0,   0,   0   }; return ret; }
    virtual double4        toDouble4() const { double4 ret = { static_cast<double       >(val), 0.0, 0.0, 0.0 }; return ret; }
    virtual char * toString() const
    {
      sprintf(canon, "ushort(%u)", val);
      return canon;
    }
};
class CudaInt : public CudaCallbackType
{
  public:
    int val;
    CudaInt(const short v = 0) : CudaCallbackType(CUDA_TYPE_INT), val(v) { }
    virtual long           toLong()    const { return static_cast<long>(val);           }
    virtual unsigned long  toULong()   const { return static_cast<unsigned long>(val);  }
    virtual double         toDouble()  const { return static_cast<double>(val);         }
    virtual void *         toPointer() const { return reinterpret_cast<void * >(val);   }
    virtual long4          toLong4()   const {  long4  ret = { static_cast<long         >(val), 0,   0,   0   }; return ret; }
    virtual ulong4         toULong4()  const { ulong4  ret = { static_cast<unsigned long>(val), 0,   0,   0   }; return ret; }
    virtual double4        toDouble4() const { double4 ret = { static_cast<double       >(val), 0.0, 0.0, 0.0 }; return ret; }
    virtual char * toString() const
    {
      sprintf(canon, "int(%d)", val);
      return canon;
    }
};
class CudaUInt : public CudaCallbackType
{
  public:
    unsigned int val;
    CudaUInt(const unsigned int v = 0) : CudaCallbackType(CUDA_TYPE_UINT), val(v) { }
    virtual long           toLong()    const { return static_cast<long>(val);          }
    virtual unsigned long  toULong()   const { return static_cast<unsigned long>(val); }
    virtual double         toDouble()  const { return static_cast<double>(val);        }
    virtual void *         toPointer() const { return reinterpret_cast<void * >(val);  }
    virtual long4          toLong4()   const {  long4  ret = { static_cast<long         >(val), 0,   0,   0   }; return ret; }
    virtual ulong4         toULong4()  const { ulong4  ret = { static_cast<unsigned long>(val), 0,   0,   0   }; return ret; }
    virtual double4        toDouble4() const { double4 ret = { static_cast<double       >(val), 0.0, 0.0, 0.0 }; return ret; }
    virtual char * toString() const
    {
      sprintf(canon, "uint(%u)", val);
      return canon;
    }
};
class CudaLong : public CudaCallbackType
{
  public:
    long val;
    CudaLong(const long v = 0) : CudaCallbackType(CUDA_TYPE_LONG), val(v) { }
    virtual long           toLong()    const { return static_cast<long>(val);          }
    virtual unsigned long  toULong()   const { return static_cast<unsigned long>(val); }
    virtual double         toDouble()  const { return static_cast<double>(val);        }
    virtual void *         toPointer() const { return reinterpret_cast<void * >(val);  }
    virtual long4          toLong4()   const {  long4  ret = { toLong(),   0,   0,   0   }; return ret; }
    virtual ulong4         toULong4()  const { ulong4  ret = { toULong(),  0,   0,   0   }; return ret; }
    virtual double4        toDouble4() const { double4 ret = { toDouble(), 0.0, 0.0, 0.0 }; return ret; }
    virtual char * toString() const
    {
      sprintf(canon, "long(%lu)", val);
      return canon;
    }
};
class CudaULong : public CudaCallbackType
{
  public:
    unsigned long val;
    CudaULong(const unsigned long v = 0) : CudaCallbackType(CUDA_TYPE_ULONG), val(v) { }
    virtual long           toLong()    const { return static_cast<long>(val);          }
    virtual unsigned long  toULong()   const { return static_cast<unsigned long>(val); }
    virtual double         toDouble()  const { return static_cast<double>(val);        }
    virtual void *         toPointer() const { return reinterpret_cast<void * >(val);  }
    virtual long4          toLong4()   const {  long4  ret = { toLong(),   0,   0,   0   }; return ret; }
    virtual ulong4         toULong4()  const { ulong4  ret = { toULong(),  0,   0,   0   }; return ret; }
    virtual double4        toDouble4() const { double4 ret = { toDouble(), 0.0, 0.0, 0.0 }; return ret; }
    virtual char * toString() const
    {
      sprintf(canon, "ulong(%lu)", val);
      return canon;
    }
};
class CudaFloat : public CudaCallbackType
{
  public:
    float val;
    CudaFloat(const float v = 0) : CudaCallbackType(CUDA_TYPE_FLOAT), val(v) { }
    virtual long           toLong()    const { return static_cast<long>(val);          }
    virtual unsigned long  toULong()   const { return static_cast<unsigned long>(val); }
    virtual float          toFloat()   const { return val;                                 }
    virtual double         toDouble()  const { return static_cast<double>(val);            }
    virtual void *         toPointer() const { return reinterpret_cast<void * >(toUInt()); }
    virtual long4          toLong4()   const {  long4  ret = { toLong(),   0,   0,   0   }; return ret; }
    virtual ulong4         toULong4()  const { ulong4  ret = { toULong(),  0,   0,   0   }; return ret; }
    virtual double4        toDouble4() const { double4 ret = { toDouble(), 0.0, 0.0, 0.0 }; return ret; }
    virtual char * toString() const
    {
      sprintf(canon, "float(%f)", val);
      return canon;
    }

};
class CudaDouble : public CudaCallbackType
{
  public:
    double val;
    CudaDouble(const double v = 0) : CudaCallbackType(CUDA_TYPE_DOUBLE), val(v) { }
    virtual long           toLong()    const { return static_cast<long>(val);          }
    virtual unsigned long  toULong()   const { return static_cast<unsigned long>(val); }
    virtual double         toDouble()  const { return val;                                 }
    virtual void *         toPointer() const { return reinterpret_cast<void * >(toUInt()); }
    virtual long4          toLong4()   const {  long4  ret = { toLong(),   0,   0,   0   }; return ret; }
    virtual ulong4         toULong4()  const { ulong4  ret = { toULong(),  0,   0,   0   }; return ret; }
    virtual double4        toDouble4() const { double4 ret = { toDouble(), 0.0, 0.0, 0.0 }; return ret; }
    virtual char * toString() const
    {
      sprintf(canon, "double(%f)", val);
      return canon;
    }
};
class CudaPointer : public CudaCallbackType
{
  public:
    void * val;
    CudaPointer(void * const v = 0) : CudaCallbackType(CUDA_TYPE_UINT), val(v) { }
    virtual long           toLong()    const { return reinterpret_cast<long>(val);          }
    virtual unsigned long  toULong()   const { return reinterpret_cast<unsigned long>(val); }
    virtual double         toDouble()  const { return static_cast<double>(toUInt());        }
    virtual void *         toPointer() const { return val; }
    virtual long4          toLong4()   const {  long4  ret = { toLong(),   0,   0,   0   }; return ret; }
    virtual ulong4         toULong4()  const { ulong4  ret = { toULong(),  0,   0,   0   }; return ret; }
    virtual double4        toDouble4() const { double4 ret = { toDouble(), 0.0, 0.0, 0.0 }; return ret; }
    virtual char * toString() const
    {
      sprintf(canon, "pointer(%p)", val);
      return canon;
    }
};
class CudaVec1 : public CudaCallbackType
{
  public:
    CudaCallbackType * e1;
    CudaVec1(CudaCallbackType * p1) : CudaCallbackType(CUDA_TYPE_INT), e1(p1) { }
    virtual ~CudaVec1() { delete e1; }
    virtual long           toLong()    const { return e1->toLong();    }
    virtual unsigned long  toULong()   const { return e1->toULong();   }
    virtual double         toDouble()  const { return e1->toDouble();  }
    virtual void *         toPointer() const { return e1->toPointer(); }
    virtual long4          toLong4()   const {  long4  ret = { e1->toLong(),   0,   0,   0   }; return ret; }
    virtual ulong4         toULong4()  const { ulong4  ret = { e1->toULong(),  0,   0,   0   }; return ret; }
    virtual double4        toDouble4() const { double4 ret = { e1->toDouble(), 0.0, 0.0, 0.0 }; return ret; }
    virtual char * toString() const
    {
      sprintf(canon, "vec1(%s)", e1->toString());
      return canon;
    }
};
class CudaVec2 : public CudaCallbackType
{
  public:
    CudaCallbackType * e1, * e2;
    CudaVec2(CudaCallbackType * p1, CudaCallbackType * p2) : CudaCallbackType(CUDA_TYPE_INT2), e1(p1), e2(p2) { }
    virtual ~CudaVec2() { delete e1; delete e2; }
    virtual long           toLong()    const { return e1->toLong();    }
    virtual unsigned long  toULong()   const { return e1->toULong();   }
    virtual double         toDouble()  const { return e1->toDouble();  }
    virtual void *         toPointer() const { return e1->toPointer(); }
    virtual long4          toLong4()   const {  long4  ret = { e1->toLong(),    e2->toLong(),    0,   0   }; return ret; }
    virtual ulong4         toULong4()  const { ulong4  ret = { e1->toULong(),   e2->toULong(),   0,   0   }; return ret; }
    virtual double4        toDouble4() const { double4 ret = { e1->toDouble(),  e2->toDouble(),  0.0, 0.0 }; return ret; }
    virtual char * toString() const
    {
      sprintf(canon, "vec2(%s,%s)", e1->toString(), e2->toString());
      return canon;
    }
};
class CudaVec3 : public CudaCallbackType
{
  public:
    CudaCallbackType * e1, * e2, * e3;
    CudaVec3(CudaCallbackType * p1, CudaCallbackType * p2, CudaCallbackType * p3) : CudaCallbackType(CUDA_TYPE_INT3), e1(p1), e2(p2), e3(p3) { }
    virtual ~CudaVec3() { delete e1; delete e2; delete e3; }
    virtual long           toLong()    const { return e1->toLong();    }
    virtual unsigned long  toULong()   const { return e1->toULong();   }
    virtual double         toDouble()  const { return e1->toDouble();  }
    virtual void *         toPointer() const { return e1->toPointer(); }
    virtual long4          toLong4()   const {  long4  ret = { e1->toLong(),    e2->toLong(),    e3->toLong(),    0   }; return ret; }
    virtual ulong4         toULong4()  const { ulong4  ret = { e1->toULong(),   e2->toULong(),   e3->toULong(),   0   }; return ret; }
    virtual double4        toDouble4() const { double4 ret = { e1->toDouble(),  e2->toDouble(),  e3->toDouble(), 0.0 }; return ret; }
    virtual char * toString() const
    {
      sprintf(canon, "vec3(%s, %s,%s)", e1->toString(), e2->toString(), e3->toString());
      return canon;
    }
};
class CudaVec4 : public CudaCallbackType
{
  public:
    CudaCallbackType * e1, * e2, * e3, * e4;
    CudaVec4(CudaCallbackType * p1, CudaCallbackType * p2, CudaCallbackType * p3, CudaCallbackType * p4)
      : CudaCallbackType(CUDA_TYPE_INT4), e1(p1), e2(p2), e3(p3), e4(p4) { }
    virtual ~CudaVec4() { delete e1; delete e2; delete e3; delete e4; }
    virtual long           toLong()    const { return e1->toLong();    }
    virtual unsigned long  toULong()   const { return e1->toULong();   }
    virtual double         toDouble()  const { return e1->toDouble();  }
    virtual void *         toPointer() const { return e1->toPointer(); }
    virtual long4          toLong4()   const {  long4  ret = { e1->toLong(),    e2->toLong(),    e3->toLong(),    e4->toLong()     }; return ret; }
    virtual ulong4         toULong4()  const { ulong4  ret = { e1->toULong(),   e2->toULong(),   e3->toULong(),   e4->toULong()    }; return ret; }
    virtual double4        toDouble4() const { double4 ret = { e1->toDouble(),  e2->toDouble(),  e3->toDouble(),  e4->toDouble()  }; return ret; }
    virtual char * toString() const
    {
      sprintf(canon, "vec4(%s,%s, %s)", e1->toString(), e2->toString(), e3->toString(), e4->toString());
      return canon;
    }
};

inline CudaCallbackType * cudaCallbackCreateValue(const cudaValue_t * val)
{
  union
  {
    char2   c2  ;
    uchar2  uc2 ;
    short2  s2  ;
    ushort2 us2 ;
    int2    i2  ;
    uint2   ui2 ;
    long2   l2  ;
    ulong2  ul2 ;
    float2  f2  ;
    double2 d2  ;

    char3   c3  ;
    uchar3  uc3 ;
    short3  s3  ;
    ushort3 us3 ;
    int3    i3  ;
    uint3   ui3 ;
    long3   l3  ;
    ulong3  ul3 ;
    float3  f3  ;
    double3 d3  ;

    char4   c4  ;
    uchar4  uc4 ;
    short4  s4  ;
    ushort4 us4 ;
    int4    i4  ;
    uint4   ui4 ;
    long4   l4  ;
    ulong4  ul4 ;
    float4  f4  ;
    double4 d4  ;
  } t;
  dim3 tdd3;

  // printf("%s.%s.%d: val->type = %s\n", __FILE__, __FUNCTION__, __LINE__, CUDA_TYPE_STRINGS[val->type]); fflush(stdout);

  switch (val->type)
  {
  case CUDA_TYPE_CHAR:      return new CudaChar   (*reinterpret_cast<volatile const char            * >(val->mem));
  case CUDA_TYPE_UCHAR:     return new CudaUChar  (*reinterpret_cast<volatile const unsigned char   * >(val->mem));
  case CUDA_TYPE_SHORT:     return new CudaShort  (*reinterpret_cast<volatile const short           * >(val->mem));
  case CUDA_TYPE_USHORT:    return new CudaUShort (*reinterpret_cast<volatile const unsigned short  * >(val->mem));
  case CUDA_TYPE_INT:       return new CudaInt    (*reinterpret_cast<volatile const int             * >(val->mem));
  case CUDA_TYPE_LONG:      return new CudaLong   (*reinterpret_cast<volatile const long            * >(val->mem));
  case CUDA_TYPE_SIZET:     return new CudaULong  (*reinterpret_cast<volatile const size_t          * >(val->mem));
  case CUDA_TYPE_UINT:      return new CudaUInt   (*reinterpret_cast<volatile const unsigned int    * >(val->mem));
  case CUDA_TYPE_ULONG:     return new CudaULong  (*reinterpret_cast<volatile const unsigned long   * >(val->mem));
  case CUDA_TYPE_FLOAT:     return new CudaFloat  (*reinterpret_cast<volatile const float           * >(val->mem));
  case CUDA_TYPE_DOUBLE:    return new CudaDouble (*reinterpret_cast<volatile const double          * >(val->mem));

  case CUDA_TYPE_CHAR2:     t.c2  = *(char2   * )val->mem; return new CudaVec2(new CudaChar   (t.c2 .x), new CudaChar   (t.c2 .y));
  case CUDA_TYPE_UCHAR2:    t.uc2 = *(uchar2  * )val->mem; return new CudaVec2(new CudaUChar  (t.uc2.x), new CudaUChar  (t.uc2.y));
  case CUDA_TYPE_SHORT2:    t.s2  = *(short2  * )val->mem; return new CudaVec2(new CudaShort  (t.s2 .x), new CudaShort  (t.s2 .y));
  case CUDA_TYPE_USHORT2:   t.us2 = *(ushort2 * )val->mem; return new CudaVec2(new CudaUShort (t.us2.x), new CudaUShort (t.us2.y));
  case CUDA_TYPE_INT2:      t.i2  = *(int2    * )val->mem; return new CudaVec2(new CudaInt    (t.i2 .x), new CudaInt    (t.i2 .y));
  case CUDA_TYPE_UINT2:     t.ui2 = *(uint2   * )val->mem; return new CudaVec2(new CudaUInt   (t.ui2.x), new CudaUInt   (t.ui2.y));
  case CUDA_TYPE_LONG2:     t.l2  = *(long2   * )val->mem; return new CudaVec2(new CudaLong   (t.l2 .x), new CudaLong   (t.l2 .y));
  case CUDA_TYPE_ULONG2:    t.ul2 = *(ulong2  * )val->mem; return new CudaVec2(new CudaULong  (t.ul2.x), new CudaULong  (t.ul2.y));
  case CUDA_TYPE_FLOAT2:    t.f2  = *(float2  * )val->mem; return new CudaVec2(new CudaFloat  (t.f2 .x), new CudaFloat  (t.f2 .y));
  case CUDA_TYPE_DOUBLE2:   t.d2  = *(double2 * )val->mem; return new CudaVec2(new CudaDouble (t.d2 .x), new CudaDouble (t.d2 .y));

  case CUDA_TYPE_CHAR3:     t.c3  = *(char3   * )val->mem; return new CudaVec3(new CudaChar   (t.c3 .x), new CudaChar   (t.c3 .y), new CudaChar   (t.c3 .z));
  case CUDA_TYPE_UCHAR3:    t.uc3 = *(uchar3  * )val->mem; return new CudaVec3(new CudaUChar  (t.uc3.x), new CudaUChar  (t.uc3.y), new CudaUChar  (t.uc3.z));
  case CUDA_TYPE_SHORT3:    t.s3  = *(short3  * )val->mem; return new CudaVec3(new CudaShort  (t.s3 .x), new CudaShort  (t.s3 .y), new CudaShort  (t.s3 .z));
  case CUDA_TYPE_USHORT3:   t.us3 = *(ushort3 * )val->mem; return new CudaVec3(new CudaUShort (t.us3.x), new CudaUShort (t.us3.y), new CudaUShort (t.us3.z));
  case CUDA_TYPE_INT3:      t.i3  = *(int3    * )val->mem; return new CudaVec3(new CudaInt    (t.i3 .x), new CudaInt    (t.i3 .y), new CudaInt    (t.i3 .z));
  case CUDA_TYPE_UINT3:     t.ui3 = *(uint3   * )val->mem; return new CudaVec3(new CudaUInt   (t.ui3.x), new CudaUInt   (t.ui3.y), new CudaUInt   (t.ui3.z));
  case CUDA_TYPE_LONG3:     t.l3  = *(long3   * )val->mem; return new CudaVec3(new CudaLong   (t.l3 .x), new CudaLong   (t.l3 .y), new CudaLong   (t.l3 .z));
  case CUDA_TYPE_ULONG3:    t.ul3 = *(ulong3  * )val->mem; return new CudaVec3(new CudaULong  (t.ul3.x), new CudaULong  (t.ul3.y), new CudaULong  (t.ul3.z));
  case CUDA_TYPE_FLOAT3:    t.f3  = *(float3  * )val->mem; return new CudaVec3(new CudaFloat  (t.f3 .x), new CudaFloat  (t.f3 .y), new CudaFloat  (t.f3 .z));
  case CUDA_TYPE_DOUBLE3:   t.d3  = *(double3 * )val->mem; return new CudaVec3(new CudaDouble (t.d3 .x), new CudaDouble (t.d3 .y), new CudaDouble (t.d3 .z));
  case CUDA_TYPE_DIM3:      tdd3  = *(dim3    * )val->mem; return new CudaVec3(new CudaUInt   (tdd3 .x), new CudaUInt   (tdd3 .y), new CudaUInt   (tdd3 .z));

  case CUDA_TYPE_CHAR4:     t.c4  = *(char4   * )val->mem; return new CudaVec4(new CudaChar   (t.c4 .x), new CudaChar   (t.c4 .y), new CudaChar   (t.c4 .z), new CudaChar   (t.c4 .w));
  case CUDA_TYPE_UCHAR4:    t.uc4 = *(uchar4  * )val->mem; return new CudaVec4(new CudaUChar  (t.uc4.x), new CudaUChar  (t.uc4.y), new CudaUChar  (t.uc4.z), new CudaUChar  (t.uc4.w));
  case CUDA_TYPE_SHORT4:    t.s4  = *(short4  * )val->mem; return new CudaVec4(new CudaShort  (t.s4 .x), new CudaShort  (t.s4 .y), new CudaShort  (t.s4 .z), new CudaShort  (t.s4 .w));
  case CUDA_TYPE_USHORT4:   t.us4 = *(ushort4 * )val->mem; return new CudaVec4(new CudaUShort (t.us4.x), new CudaUShort (t.us4.y), new CudaUShort (t.us4.z), new CudaUShort (t.us4.w));
  case CUDA_TYPE_INT4:      t.i4  = *(int4    * )val->mem; return new CudaVec4(new CudaInt    (t.i4 .x), new CudaInt    (t.i4 .y), new CudaInt    (t.i4 .z), new CudaInt    (t.i4 .w));
  case CUDA_TYPE_UINT4:     t.ui4 = *(uint4   * )val->mem; return new CudaVec4(new CudaUInt   (t.ui4.x), new CudaUInt   (t.ui4.y), new CudaUInt   (t.ui4.z), new CudaUInt   (t.ui4.w));
  case CUDA_TYPE_LONG4:     t.l4  = *(long4   * )val->mem; return new CudaVec4(new CudaLong   (t.l4 .x), new CudaLong   (t.l4 .y), new CudaLong   (t.l4 .z), new CudaLong   (t.l4 .w));
  case CUDA_TYPE_ULONG4:    t.ul4 = *(ulong4  * )val->mem; return new CudaVec4(new CudaULong  (t.ul4.x), new CudaULong  (t.ul4.y), new CudaULong  (t.ul4.z), new CudaULong  (t.ul4.w));
  case CUDA_TYPE_FLOAT4:    t.f4  = *(float4  * )val->mem; return new CudaVec4(new CudaFloat  (t.f4 .x), new CudaFloat  (t.f4 .y), new CudaFloat  (t.f4 .z), new CudaFloat  (t.f4 .w));
  case CUDA_TYPE_DOUBLE4:   t.d4  = *(double4 * )val->mem; return new CudaVec4(new CudaDouble (t.d4 .x), new CudaDouble (t.d4 .y), new CudaDouble (t.d4 .z), new CudaDouble (t.d4 .w));

  default:
    // printf("%s.%s.%d: Creating new pointer out of *%p=%p\n", __FILE__, __FUNCTION__, __LINE__, val->mem, *(void ** )val->mem); fflush(stdout);
    return new CudaPointer(*(void ** )val->mem);
  }
}

#endif
