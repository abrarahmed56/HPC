#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void vector_dot_product(double* dot_product_ptr, const double* a, const double* b, long N){
  double sum = 0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++) sum += a[i] * b[i];
  *dot_product_ptr = sum;
}

void matrix_vector_product(double* prod_pts, const double* v, const double* M, long N) {
  //vector_dot_product()
  for (long i = 0; i < N; i++) {
    double prod = 0;
    vector_dot_product(&prod, v, &(M[i*N]), N);
    *(prod_pts+i) = prod;
  }
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

#define BLOCK_SIZE 1024

// Warp divergence
__global__ void reduction_kernel0(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x %   2 == 0) smem[threadIdx.x] += smem[threadIdx.x + 1];
  __syncthreads();
  if (threadIdx.x %   4 == 0) smem[threadIdx.x] += smem[threadIdx.x + 2];
  __syncthreads();
  if (threadIdx.x %   8 == 0) smem[threadIdx.x] += smem[threadIdx.x + 4];
  __syncthreads();
  if (threadIdx.x %  16 == 0) smem[threadIdx.x] += smem[threadIdx.x + 8];
  __syncthreads();
  if (threadIdx.x %  32 == 0) smem[threadIdx.x] += smem[threadIdx.x + 16];
  __syncthreads();
  if (threadIdx.x %  64 == 0) smem[threadIdx.x] += smem[threadIdx.x + 32];
  __syncthreads();
  if (threadIdx.x % 128 == 0) smem[threadIdx.x] += smem[threadIdx.x + 64];
  __syncthreads();
  if (threadIdx.x % 256 == 0) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x % 512 == 0) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x] + smem[threadIdx.x + 512];
}

// Shared memory bank conflicts
__global__ void reduction_kernel1(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x *   2] += smem[threadIdx.x *   2 +   1];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x *   4] += smem[threadIdx.x *   4 +   2];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x *   8] += smem[threadIdx.x *   8 +   4];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x *  16] += smem[threadIdx.x *  16 +   8];
  __syncthreads();
  if (threadIdx.x <  32) smem[threadIdx.x *  32] += smem[threadIdx.x *  32 +  16];
  __syncwarp();
  if (threadIdx.x <  16) smem[threadIdx.x *  64] += smem[threadIdx.x *  64 +  32];
  __syncwarp();
  if (threadIdx.x <   8) smem[threadIdx.x * 128] += smem[threadIdx.x * 128 +  64];
  __syncwarp();
  if (threadIdx.x <   4) smem[threadIdx.x * 256] += smem[threadIdx.x * 256 + 128];
  __syncwarp();
  if (threadIdx.x <   2) smem[threadIdx.x * 512] += smem[threadIdx.x * 512 + 256];
  __syncwarp();
  if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[512];
}

__global__ void reduction_kernel2(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

__global__ void matrix_vector_product_kernel2(double* sum, const double* a, const double* b, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N*N) smem[threadIdx.x] = a[idx%N] * b[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

int main() {
  //long N = (1UL<<25);
  // changed N to prevent segmentation fault due to insufficient stack space
  long N = (1UL<<15);

  double *v;
  cudaMallocHost((void**)&v, N * sizeof(double));
  //double* M = (double*) (malloc(N * N * sizeof(double)));
  double *M;
  cudaMallocHost((void**)&M, N * N * sizeof(double)); //N x N matrix
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    v[i] = 1.0/(i+1);
  }
  for (long i = 0; i < N * N; i++) {
    M[i] = 1.0/(i+1);
  }

  //double prod_ref, sum;
  double* prod_ref = (double*) (malloc(N * sizeof(double)));
  //double* sum = (double*) (malloc(N * sizeof(double)));
  double* sum;
  cudaMallocHost(&sum, N*sizeof(double));
  double tt = omp_get_wtime();
  //reduction(&sum_ref, x, N);
  //vector_dot_product(&dot_product_ref, x1, x2, N);
  matrix_vector_product(prod_ref, v, M, N);
  printf("CPU Bandwidth = %f GB/s\n", 1*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double *v_d, *M_d, *y_d;
  cudaMalloc(&v_d, N*sizeof(double));
  cudaMalloc(&M_d, N*N*sizeof(double));
  long N_work = 1;
  for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  cudaMalloc(&y_d, N_work*sizeof(double)); // extra memory buffer for reduction across thread-blocks

  cudaMemcpyAsync(v_d, v, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(M_d, M, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  tt = omp_get_wtime();


  double* sum_d = y_d;
  long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  matrix_vector_product_kernel2<<<Nb,BLOCK_SIZE>>>(sum_d, v_d, M_d, N);
  while (Nb > 1) {
    long N = Nb;
    Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
    reduction_kernel2<<<Nb,BLOCK_SIZE>>>(sum_d + N, sum_d, N);
    sum_d += N;
  }


  cudaMemcpyAsync(sum, sum_d, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", 1*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  double diff = 0;
  double sum_sum = 0;
  double prod_ref_sum = 0;
  for (long i = 0; i < N; i++) {
    diff += fabs(sum[i]-prod_ref[i]);
    sum_sum += sum[i];
    prod_ref_sum += prod_ref[i];
  }
  printf("Error = %f\n", diff);
  printf("Sum sum = %f\n", sum_sum);
  printf("Prod ref sum = %f\n", prod_ref_sum);

  cudaFree(v_d);
  cudaFree(M_d);
  cudaFree(y_d);

  cudaFreeHost(v);
  cudaFreeHost(M);
  cudaFreeHost(sum);

  return 0;
}