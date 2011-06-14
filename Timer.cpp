#ifdef _WIN32
  #include <windows.h>
#else
  #include <sys/time.h>
#endif

#include <Timer.h>
#include <cstdio>

Timer globalTimer;

#ifdef _WIN32

// inline static LARGE_INTEGER & licast(void * const p) { return *reinterpret_cast<LARGE_INTEGER * >(p); }
inline static LARGE_INTEGER & licast(void * const p) { return *reinterpret_cast<LARGE_INTEGER * >(p); }

  void setAffinity()
  {
    if (SetProcessAffinityMask(GetCurrentProcess(), 1) == 0)
    {
      DWORD error = GetLastError();
      printf("spam failed: %d\n", static_cast<int>(error));
      fflush(stdout);
      exit(1);
    }
  }

  Timer::Timer()
  {
    begin = data;
    end = data + sizeof(LARGE_INTEGER);
    misc = data + sizeof(LARGE_INTEGER) * 2;
    QueryPerformanceFrequency(&licast(misc));
  }

  void Timer::start()
  {
    QueryPerformanceCounter(&licast(begin));
  }
  void Timer::stop()
  {
    QueryPerformanceCounter(&licast(end));
  }
  double Timer::getElapsedSeconds() const
  {
    double diff = static_cast<double>(licast(end).QuadPart - licast(begin).QuadPart);
    return diff / static_cast<double>(licast(misc).QuadPart);
  }

#else

  inline static struct timeval & tvcast(void * const p) { return *reinterpret_cast<struct timeval * >(p); }

  Timer::Timer()
  {
    started = false;
    stopped = false;
    begin = data;
    end = data + sizeof(struct timeval);
    misc = data + sizeof(struct timeval);
    start();
  }
  void Timer::start()
  {
    gettimeofday(&tvcast(begin), NULL);
    started = true;
    stopped = false;
  }
  void Timer::stop()
  {
    gettimeofday(&tvcast(end), NULL);
    started = false;
    stopped = true;
  }
  double Timer::getElapsedSeconds() const
  {
    if (!stopped) gettimeofday(&tvcast(end), NULL);
    double s0 = tvcast(begin).tv_sec + tvcast(begin).tv_usec / 1000000.0;
    double s1 = tvcast(end  ).tv_sec + tvcast(end  ).tv_usec / 1000000.0;
    return s1 - s0;
  }
#endif

double Timer::getElapsedMilliseconds() const
{
  return getElapsedSeconds() * 1000.0;
}
double Timer::getElapsedMicroseconds() const
{
  return getElapsedSeconds() * 1000000.0;
}
