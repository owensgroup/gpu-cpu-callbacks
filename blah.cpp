#include <cstdio>

template <typename T1> int typesize() { return sizeof(T1); }

int main(int argc, char ** argv)
{
  printf("typesize<int>(): %d\n", typesize<int>());
  return 0;
}
