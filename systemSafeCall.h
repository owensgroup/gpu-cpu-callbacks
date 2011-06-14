#ifndef __SYSTEMSAFECALL_H__
#define __SYSTEMSAFECALL_H__

#include <cerrno>

// A nice wrapper to make sure that CUDA calls are executing properly.

#define SYSTEM_SAFE_CALL(x)                                                                             \
{                                                                                                       \
  errno = 0;                                                                                            \
  printf("%s.%s.%d: %s\n", __FILE__, __FUNCTION__, __LINE__, #x);                                       \
  (x);                                                                                                  \
  if (errno != 0)                                                                                       \
  {                                                                                                     \
    printf("%s.%s.%d: '%s' 0x%x (%s)\n", __FILE__, __FUNCTION__, __LINE__, #x, errno, strerror(errno)); \
    exit(1);                                                                                            \
  }                                                                                                     \
}                                                                                                       \

#endif
