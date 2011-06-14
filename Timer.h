#ifndef __TIMER_H__
#define __TIMER_H__

void setAffinity();

// Just a basic class used for timing. Works cross platform.
class Timer
{
  protected:
    void * begin, * end, * misc;
    bool started, stopped;
    char data[128];
  public:
    Timer();

    inline bool isRunning() const { return started && !stopped; }

    void start();
    void stop();

    double getElapsedSeconds() const;
    double getElapsedMilliseconds() const;
    double getElapsedMicroseconds() const;
};

extern Timer globalTimer;

#endif
