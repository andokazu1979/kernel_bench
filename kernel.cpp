#include "kernel.h"

int Kernel::get_dim() {
  return dim;
}

int Kernel::get_nflop() {
  return nflop;
}

KernelMVM::KernelMVM() {
  cout << "constructor of KernelMVM" << endl;
  dim = 2;
  nflop = 2;
}

void KernelMVMClassic::init(int n) {
  m = new double*[n];
  v = new double[n];
  rv = new double[n];

  for(int i = 0; i < n; i++) {
    m[i] = new double[n];
    rv[i] = 0.0f;
#ifdef DEBUG
    v[i] = 1.0f;
    for(int j = 0; j < n; j++) {
      m[i][j] = (double)(i * n + j);
    }
#else
    v[i] = (double)rand();
    for(int j = 0; j < n; j++) {
      m[i][j] = (double)rand();
    }
#endif
  } 
}

void KernelMVMClassic::calc(int n_, int nitr_) {
  for(int itr = 0; itr < nitr_; itr++) {
    //#pragma omp parallel for
    for(int i = 0; i < n_; i++) {
      for(int j = 0; j < n_; j++) {
        rv[i] += m[i][j] * v[j];
      }
    }
    if(rv[n_-1] < 0) dummy(rv);
  }
}

void KernelMVMClassic::fin(int n) {
  for(int i = 0; i < n; i++) {
    delete[] m[i];
  }
  delete[] m;
  delete[] v;
  delete[] rv;
}

#ifdef CBLAS
void KernelMVMCblas::init(int n) {
  v = new double[n];
  rv = new double[n];
  m = new double[n*n];

  order = CblasRowMajor;
  trans = CblasNoTrans;
  //trans = CblasTrans;
  alpha = 1.0;
  beta = 0.0;
  lda = n;
  incx = 1;
  incy = 1;

  for(int i = 0; i < n; i++) {
    rv[i] = 0.0;
#ifdef DEBUG
    v[i] = 1.0;
    for(int j = 0; j < n; j++) {
      m[n*i+j] = (double)(i * n + j);
    }
#else
    v[i] = (double)rand();
    for(int j = 0; j < n; j++) {
      m[n*i+j] = (double)rand();
    }
#endif
  }
}

void KernelMVMCblas::calc(int n_, int nitr_) {
  for(int itr = 0; itr < nitr_; itr++) {
    cblas_dgemv(order, trans, n_, n_, alpha, m, lda, v, incx, beta, rv, incy);
    if(rv[n_-1] < 0) dummy(rv);
  }
}

void KernelMVMCblas::fin(int n) {
  delete[] v;
  delete[] rv;
  delete[] m;
}
#endif

long gettimeofday_usec() {
  struct timeval tv;
  gettimeofday(&tv,  NULL);
  return tv.tv_sec*1e+6 + tv.tv_usec;
}

void dummy(double* val) {
}

#ifdef EIGEN
void KernelMVMEigen::init(int n) {
#ifdef DEBUG
  m(0, 0) = 0.0;
  m(0, 1) = 1.0;
  m(1, 0) = 2.0;
  m(1, 1) = 3.0;
  v = VectorXd::Constant(n, 1.0);
  rv = VectorXd::Constant(n, 0.0);
#else
  m = MatrixXd::Random(n, n);
  v = VectorXd::Random(n);
  rv = VectorXd::Constant(n, 0.0);
#endif
}

void KernelMVMEigen::calc(int n_, int nitr_) {
  for(int itr = 0; itr < nitr_; itr++) {
    rv = m * v;
    if(rv[n_-1] < 0) dummy(rv);
  }
}

void KernelMVMEigen::fin(int n) {
}

void dummy(Eigen::VectorXd val) {
}
#endif

KernelScaling::KernelScaling() {
  cout << "constructor of KernelScaling" << endl;
  dim = 1;
  nflop = 1;
}

void KernelScaling::init(int n) {
  v = new double[n];

  scale = (double)rand();

  for(int i = 0; i < n; i++) {
    v[i] = (double)rand();
  } 
}

void KernelScaling::fin(int n) {
  delete[] v;
}

void KernelScalingClassic::calc(int n_, int nitr_) {
  for(int itr = 0; itr < nitr_; itr++) {
    //#pragma omp parallel for
    for(int i = 0; i < n_; i++) {
      v[i] = scale * v[i];
    }
    if(v[n_-1] < 0) dummy(v);
  }
}

KernelAdd::KernelAdd() {
  cout << "constructor of KernelAdd" << endl;
  dim = 1;
  nflop = 1;
}

void KernelAddClassic::init(int n) {
  v1 = new double[n];
  v2 = new double[n];
  v3 = new double[n];

  for(int i = 0; i < n; i++) {
    v1[i] = (double)rand();
    v2[i] = (double)rand();
    v3[i] = 0.0f;
  } 
}

void KernelAddClassic::calc(int n_, int nitr_) {
  for(int itr = 0; itr < nitr_; itr++) {
    for(int i = 0; i < n_; i++) {
      v3[i] = v1[i] + v2[i];
    }
    if(v3[n_-1] < 0) dummy(v3);
  }
}

void KernelAddClassic::fin(int n) {
  delete[] v1;
  delete[] v2;
  delete[] v3;
}

