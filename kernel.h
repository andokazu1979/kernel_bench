#ifndef KERNEL_H
#define KERNEL_H

#include <iostream>
#include <cstdlib>
#include <sys/time.h>

#ifdef EIGEN
#include <Eigen/Dense>
#endif

#ifdef CBLAS
#include "mkl_cblas.h"
#endif

#include "timer.h"

using namespace std;
#ifdef EIGEN
using namespace Eigen;
#endif

long gettimeofday_usec();
void dummy(double* val);
#ifdef EIGEN
void dummy(Eigen::VectorXd val);
#endif

class Kernel {
public:
  void doProcess();
  virtual void init(int n);
  virtual void calc(int n_, int nitr_);
  virtual void fin(int n);
  int get_dim();
  int get_nflop();
protected:
  int dim;
  int nflop;
  Timer* timer;
};

class KernelMVM : public Kernel {
public:
  KernelMVM();
  void init(int n);
  //void calc(int n_, int nitr_);
};

class KernelMVMClassic : public KernelMVM {
  double** m;
  double* v;
  double* rv;
public:
  void init(int n);
  void calc(int n_, int nitr_);
  void fin(int n);
};

#ifdef CBLAS
class KernelMVMCblas : public KernelMVM {
  double* m;
  double* v;
  double* rv;
  CBLAS_ORDER     order;
  CBLAS_TRANSPOSE trans;
  double          alpha;
  double          beta;
  int             lda;
  int             incx;
  int             incy;
public:
  void init(int n);
  void calc(int n_, int nitr_);
  void fin(int n);
};
#endif

#ifdef EIGEN
class KernelMVMEigen : public KernelMVM {
  MatrixXd m;
  VectorXd v;
  VectorXd rv;
public:
  void init(int n);
  void calc(int n_, int nitr_);
  void fin(int n);
};
#endif

class KernelScaling : public Kernel {
protected:
  double scale;
  double* v;
public:
  KernelScaling();
  void init(int n);
  void fin(int n);
};

class KernelScalingClassic : public KernelScaling {
public:
  void calc(int n_, int nitr_);
};


#endif
