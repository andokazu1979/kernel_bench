#include "kernel_bench.h"

using namespace std;

int main(int argc, char* argv[]) {
  //KernelBench kernelbench(new KernelMVMEigen(), new TimerGetTimeOfDayUsec(), 30000);
  KernelBench kernelbench(new KernelScalingClassic(), new TimerGetTimeOfDayUsec(), 30000);

  kernelbench.doProcess();

  return 0;
}

KernelBench::KernelBench(Kernel* kernel_, Timer* timer_, int nmax) {
  cout.setf(ios::scientific,  ios::floatfield);

  kernel = kernel_;
  timer = timer_;

  size = nmax / 10;
  n = new int[size];
  nitr = new int[size];

  for(int i = 0; i < size; i++) {
    n[i] = 10 + i * 10;
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

