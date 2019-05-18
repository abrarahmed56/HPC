#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

long NUM_ITERATIONS = 100;
long N = 5000;
#define BLOCK_SIZE 1024

void jacobi(double* u, double* f) {
    int count = 0;
    //int sum_not_changed = 0;
    //float old_residual;
    
    while (count < NUM_ITERATIONS) {
      double* new_u = (double*) malloc(N * N * sizeof(double));
      #pragma omp parallel for
      for (int j = N - 1; j > 0; j--) {
	for (int i = 1; i < N; i++) {
	  float new_value = (f[i*N + j]/((N+1)*(N+1)) + u[(i-1)*N + j] + u[i*N + j-1] + u[(i+1)*N + j] + u[i*N + j+1]) / 4;
	  new_u[i*N + j] = new_value;
	}
      }

      memcpy(u, new_u, N*N*sizeof(double));
      free(new_u);
      
      count++;
    }
}

__global__ void jacobi_kernel(double* u, double* f, double* new_u, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N * N) {
       new_u[idx] = (f[idx]/((N+1)*(N+1)) + u[idx-N] + u[idx-1] + u[idx+N] + u[idx+1]) / 4;
    }
}

void jacobi_gpu(double* u, double* f) {
    int count = 0;
    int num_blocks = N / BLOCK_SIZE + 1;

    while (count < NUM_ITERATIONS) {
      	  double* new_u = (double*) malloc(N * N * sizeof(double));
    	  jacobi_kernel<<<num_blocks, BLOCK_SIZE>>>(u, f, new_u, N);
      	  memcpy(u, new_u, N*N*sizeof(double));
      	  free(new_u);
	  count++;
    }
}

int main(int argc, char** argv) {
  double* u = (double*) malloc(N * N * sizeof(double));
  double* f = (double*) malloc(N * N * sizeof(double));
  for (long i = 0; i < N; i++) {
    for (long j = 0; j < N; j++) {
      u[i*N+j] = 0;
      f[i*N+j] = 1;
    }
  }

  Timer t;

  t.tic();
  jacobi(u, f);
  printf("Serial Jacobi time: %f s\n", t.toc());
  
  t.tic();
  jacobi_gpu(u, f);
  printf("GPU Jacobi time: %f s\n", t.toc());

  free(u);
  free(f);
  return 0;
}
