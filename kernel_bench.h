#ifndef KERNEL_BENCH_H
#define KERNEL_BENCH_H

#include <iostream>
#include <cmath>

#include "kernel.h"
#include "timer.h"

class KernelBench {
  Kernel* kernel;
  Timer* timer;
  int size;
  int* n;
  int* nitr;
public:
  void doProcess();
  KernelBench(Kernel* kernel_, Timer* timer_, int nmax);
private:
  void print(int count);
};

#endif
