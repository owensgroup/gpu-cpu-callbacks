#include <cstdlib>
#include <cstdio>

int main(int argc, char ** argv)
{
  size_t s;
  FILE * fp = fopen(argv[1], "wb");
  sscanf(argv[2], "%u", &s);
  char * buf = new char[s];
  for (size_t i = 0; i < s; ++i) buf[i] = rand();
  fwrite(buf, s, 1, fp);
  fclose(fp);
  return 0;
}
