#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>
#include <stddef.h>

class Timer {
public:
  //double elapse;
  //double mflops;
  virtual void start();
  virtual void end();
  virtual double get_elapse();
  virtual double get_mflops(int nflop);
};

class TimerGetTimeOfDay : public Timer {
public:
  virtual void start();
  virtual void end();
  virtual double get_elapse();
  virtual double get_mflops(int nflop);
};

class TimerGetTimeOfDaySec : public TimerGetTimeOfDay {
  double start_;
  double end_;
public:
  virtual void start();
  virtual void end();
  virtual double get_elapse();
  virtual double get_mflops(int nflop);
private:
  long gettimeofday_sec();
};

class TimerGetTimeOfDayUsec : public TimerGetTimeOfDay {
  long start_;
  long end_;
public:
  virtual void start();
  virtual void end();
  virtual double get_elapse();
  virtual double get_mflops(int nflop);
private:
  long gettimeofday_usec();
};

#endif
