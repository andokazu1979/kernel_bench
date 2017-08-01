#include "kernel_bench.h"

using namespace std;

int main(int argc, char* argv[]) {
  KernelBench* kernelbench = NULL;
  switch (atoi(argv[1])) {
  case 1:
    cout << "MVM Classic" << endl;
    kernelbench = new KernelBench(new KernelMVMClassic(),       new TimerGetTimeOfDayUsec(), 3000);
    break;
  case 2:
    cout << "MVM CBLAS" << endl;
    kernelbench = new KernelBench(new KernelMVMCblas(),         new TimerGetTimeOfDayUsec(), 3000);
    break;
  case 3:
    cout << "Scaling Classic" << endl;
    kernelbench = new KernelBench(new KernelScalingClassic(),   new TimerGetTimeOfDayUsec(), 300000);
    break;
  case 4:
    cout << "Vector Add Classic" << endl;
    kernelbench = new KernelBench(new KernelAddClassic(),       new TimerGetTimeOfDayUsec(), 300000);
    break;
  case 5:
    cout << "Vector Add Cuda method1" << endl;
    kernelbench = new KernelBench(new KernelAddCuda(),          new TimerGetTimeOfDayUsec(), 300000);
    break;
  case 6:
    cout << "Vector Add Cuda method2" << endl;
    kernelbench = new KernelBench(new KernelAddCuda2(),         new TimerGetTimeOfDayUsec(), 300000);
    break;
  case 7:
    cout << "2D Jacobi algorithm Classic" << endl;
    kernelbench = new KernelBench(new Kernel2DJacobiClassic(),  new TimerGetTimeOfDayUsec(), 30);
    break;
  }

  kernelbench->doProcess();
  kernelbench->check(4);

  return 0;
}

KernelBench::KernelBench(Kernel* kernel_, Timer* timer_, int nmax) {
  cout.setf(ios::scientific,  ios::floatfield);

  kernel = kernel_;
  timer = timer_;
  //int interval = 10000;
  int interval = nmax / 30;

  size = nmax / interval;
  n = new int[size];
  nitr = new int[size];

  for(int i = 0; i < size; i++) {
    n[i] = interval + i * interval;
    nitr[i] = pow((double)nmax / (double)n[i], kernel->get_dim());
  }
}

void KernelBench::doProcess() {
  for(int count = 0; count < size; count++) {
    kernel->init(n[count]);
    
    timer->start();
    
    kernel->calc(n[count], nitr[count]);

    timer->end();

    kernel->fin(n[count]);

    print(count);
  } 
}

void KernelBench::print(int count) {
  if(count == 0) cout << "n,nitr,elapse(s),MFLOPS" << endl;
  cout << n[count] << "," << nitr[count] << "," << timer->get_elapse() << "," 
    //<< timer->get_mflops(2 * nitr[count] * pow(n[count], kernel->get_dim())) << endl;
    << timer->get_mflops(kernel->get_nflop() * nitr[count] * pow(n[count], kernel->get_dim())) << endl;
}

void KernelBench::check(int n) {
    kernel->init(n);
    kernel->calc(n, 1);
    cout << "result:" << endl;
    kernel->show(n);
    kernel->fin(n);
}

