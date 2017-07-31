#ifdef CUDA

#include "kernel.h" 
//#include "kernel_cuda.h" 

__global__ void add(int n_, double* v1, double* v2, double* v3) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n_; i += stride) {
    v3[i] = v1[i] + v2[i];
  }
}

__global__ void add2(int n_, double* v1, double* v2, double* v3) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < n_) v3[index] = v1[index] + v2[index];
}

void KernelAddCuda::init(int n) {
  cudaMallocManaged(&v1, n*sizeof(double)); 
  cudaMallocManaged(&v2, n*sizeof(double)); 
  cudaMallocManaged(&v3, n*sizeof(double)); 

  for(int i = 0; i < n; i++) {
    //v1[i] = (double)rand();
    v1[i] = (double)i;
    //v2[i] = (double)rand();
    v2[i] = (double)i;
    v3[i] = 0.0f;
  } 
}

void KernelAddCuda::calc(int n_, int nitr_) {
  for(int itr = 0; itr < nitr_; itr++) {
    add<<<1, 256>>>(n_, v1, v2, v3);
    cudaDeviceSynchronize();
    if(v3[n_-1] < 0) dummy(v3);
  }
}

void KernelAddCuda::fin(int n) {
  cudaFree(v1);
  cudaFree(v2);
  cudaFree(v3);
}

void KernelAddCuda2::init(int n) {
  cudaMallocManaged(&v1, n*sizeof(double)); 
  cudaMallocManaged(&v2, n*sizeof(double)); 
  cudaMallocManaged(&v3, n*sizeof(double)); 

  for(int i = 0; i < n; i++) {
    v1[i] = (double)i;
    v2[i] = (double)i;
    v3[i] = 0.0;
  }
}

void KernelAddCuda2::calc(int n_, int nitr_) {
  dim3 block(n_);
  dim3 grid((n_ + block.x - 1) / block.x);
  for(int itr = 0; itr < nitr_; itr++) {
    add2<<<grid, block>>>(n_, v1, v2, v3);
    cudaDeviceSynchronize();
    if(v3[n_-1] < 0) dummy(v3);
  }
}

void KernelAddCuda2::fin(int n) {
  cudaFree(v1);
  cudaFree(v2);
  cudaFree(v3);
}
#endif
