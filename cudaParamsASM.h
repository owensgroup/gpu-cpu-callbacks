#ifndef __CUDAPARAMSASM_H__
#define __CUDAPARAMSASM_H__

// Not used. Eventually I wanted to be able to call a function from the callback
// library using assembly so we wouldn't need to marshall parameters. Alas,
// another time.

#define CUDA_CALL(x) __asm call x

#define CUDA_PUSH_CHAR(c)       \
__asm mov   eax, dword ptr [c]  \
__asm mov   cl, byte ptr [eax]  \
__asm push  ecx                 \

#define CUDA_PUSH_SHORT(s)      \
__asm mov   eax, dword ptr [s]  \
__asm mov   cx, word ptr [eax]  \
__asm push  ecx                 \

#define CUDA_PUSH_INT(i)        \
__asm mov eax, dword ptr [i]    \
__asm mov ecx, dword ptr [eax]  \
__asm push ecx                  \

#define CUDA_PUSH_LONG(l)       \
CUDA_PUSH_INT(l)                \

#define CUDA_PUSH_SIZET(l)      \
CUDA_PUSH_INT(l)                \

#define CUDA_PUSH_FLOAT(f)      \
__asm mov eax, dword ptr [dp]   \
__asm push ecx                  \
__asm fld dword ptr [eax]       \
__asm fstp dword ptr [esp]      \

#define CUDA_PUSH_DOUBLE(f)     \
__asm mov eax, dword ptr [f]    \
__asm sub esp, 8                \
__asm fld qword ptr [eax]       \
__asm fstp qword ptr [esp]      \

#define CUDA_PUSH_CHAR1(c)      \
CUDA_PUSH_CHAR(c)               \

#define CUDA_PUSH_CHAR2(c)      \
CUDA_PUSH_SHORT(c)              \

#define CUDA_PUSH_CHAR3(c)      \

#define CUDA_GET_CHAR(c)        \
__asm mov ecx, dword ptr [c]    \
__asm mov byte ptr [ecx], al    \

#define CUDA_GET_SHORT(c)       \
__asm mov ecx, dword ptr [c]    \
__asm mov word ptr [ecx], ax    \

#define CUDA_GET_INT(c)         \
__asm mov ecx, dword ptr [c]    \
__asm mov dword ptr [ecx], eax  \

#define CUDA_GET_FLOAT(f)       \
__asm mov eax, f                \
__asm fstp dword ptr [eax]      \

#define CUDA_GET_DOUBLE(f)      \
__asm mov eax, f                \
__asm fstp qword ptr [eax]      \

#define CUDA_GET_CHAR1(c)       \
__asm mov edx, c                \
__asm mov byte ptr [edx], al    \

#define CUDA_GET_CHAR2(c)       \
__asm mov edx, c                \
__asm mov word ptr [edx], ax    \

#define CUDA_MOVE_STACK_PTR(amt) __asm add esp, amt


#define CUDA_PUSH_UCHAR(c)    CUDA_PUSH_CHAR(c)
#define CUDA_PUSH_USHORT(c)   CUDA_PUSH_SHORT(c)
#define CUDA_PUSH_UINT(c)     CUDA_PUSH_INT(c)
#define CUDA_PUSH_ULONG(c)    CUDA_PUSH_LONG(c)

#endif
