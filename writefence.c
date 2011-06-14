#include <xmmintrin.h>

void __writefence() {
  _mm_sfence();
}
