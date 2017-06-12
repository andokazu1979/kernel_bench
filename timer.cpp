#include "timer.h"

void Timer::start() {
}

void Timer::end() {
}

double Timer::get_elapse() {
  return 0.0;
}

double Timer::get_mflops(int nflop) {
  return 0.0;
}

void TimerGetTimeOfDay::start() {
}

void TimerGetTimeOfDay::end() {
}

double TimerGetTimeOfDay::get_elapse() {
  return 0.0;
}

double TimerGetTimeOfDay::get_mflops(int nflop) {
  return 0.0;
}

long TimerGetTimeOfDaySec::gettimeofday_sec() {
  struct timeval tv;
  gettimeofday(&tv,   NULL);
  return (double)(tv.tv_sec + tv.tv_usec/1000000.0);
}

void TimerGetTimeOfDaySec::start() {
}

void TimerGetTimeOfDaySec::end() {
}

double TimerGetTimeOfDaySec::get_elapse() {
  return 0.0;
}

double TimerGetTimeOfDaySec::get_mflops(int nflop) {
  return 0.0;
}

long TimerGetTimeOfDayUsec::gettimeofday_usec() {
  struct timeval tv;
  gettimeofday(&tv,  NULL);
  return tv.tv_sec*1e+6 + tv.tv_usec;
}

void TimerGetTimeOfDayUsec::start() {
  start_ = gettimeofday_usec();
}

void TimerGetTimeOfDayUsec::end() {
  end_ = gettimeofday_usec();
}

double TimerGetTimeOfDayUsec::get_elapse() {
  return (double)(end_ - start_) * 1.0e-6;
}

double TimerGetTimeOfDayUsec::get_mflops(int nflop) {
  return nflop / (get_elapse() * 1000000.0);
}
