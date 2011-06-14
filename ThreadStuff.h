#ifndef __THREADSTUFF_H__
#define __THREADSTUFF_H__

// A bunch of high-level threading classes. Nothing really special to see.

#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <cstdio>

class Thread;
class Runner;
class Condition;
class Lock;

class Runner
{
  public:
    inline Runner() { }
    inline virtual ~Runner() { }

    virtual void run() = 0;
};

class Thread
{
  protected:
    pthread_t tid;
    volatile bool running;
    Runner * runner;

    inline static void * startThread(void * t)
    {
      Thread * thread = static_cast<Thread * >(t);
      thread->runner->run();
      thread->running = false;
      return NULL;
    }
  public:
    inline Thread(Runner * const pRunner) : running(false), runner(pRunner)
    {
    }
    virtual ~Thread()
    {
      if (running)
      {
        fprintf(stderr, "Warning, thread deleted while still running.\n");
        fflush(stderr);
      }
      if (runner) delete runner;
    }
    inline bool isRunning() const { return running; }
    inline pthread_t & getNativeHandle() { return tid; }
    inline void start()
    {
      if (runner == NULL) return;
      if (running) return;
      running = true;
      pthread_create(&tid, NULL, startThread, this);
    }
    inline void join()
    {
      if (!running) return;
      pthread_join(tid, NULL);
    }
};
class Condition
{
  protected:
    pthread_cond_t handle;
    pthread_mutex_t * mutex;
  public:
    inline Condition(pthread_mutex_t * const pMutex) : mutex(pMutex)
    {
      pthread_cond_init(&handle, NULL);
    }
    inline ~Condition()
    {
      pthread_cond_destroy(&handle);
    }
    inline void wait()
    {
      pthread_cond_wait(&handle, mutex);
    }
    inline void signal()
    {
      pthread_cond_signal(&handle);
    }
    inline void broadcast()
    {
      pthread_cond_broadcast(&handle);
    }
};
class Lock
{
  protected:
    pthread_mutex_t handle;
  public:
    inline Lock()
    {
      pthread_mutex_init(&handle, NULL);
    }
    inline ~Lock()
    {
      pthread_mutex_destroy(&handle);
    }
    inline Condition * newCondition()
    {
      return new Condition(&handle);
    }
    inline void lock()
    {
      pthread_mutex_lock(&handle);
    }
    inline void unlock()
    {
      pthread_mutex_unlock(&handle);
    }
    inline bool tryLock() // return true iff we locked lock with this call.
    {
      return pthread_mutex_unlock(&handle) != EBUSY;
    }
};

#endif
