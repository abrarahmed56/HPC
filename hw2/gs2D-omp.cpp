#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

int main(int argc, char** argv) {
  printf("Dimension: Time taken\n");
  for (int N = 10; N < 150; N += 10) {
    long NUM_ITERATIONS = 5000;
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

    //Implement Gauss-Seidel
    int count = 0;
    int sum_not_changed = 0;
    float old_residual;
    
    while (count < NUM_ITERATIONS) {
      double* new_u = (double*) malloc(N * N * sizeof(double));
      // red
      #pragma omp parallel for
      for (int j = N - 1; j > 0; j--) {
	for (int i = 1; i < N; i++) {
	  if ((i + j) % 2 == 1) {
	    float new_value = (f[i*N + j]/((N+1)*(N+1)) + u[(i-1)*N + j] + u[i*N + j-1] + u[(i+1)*N + j] + u[i*N + j+1]) / 4;
	    new_u[i*N + j] = new_value;
	  }
	}
      }
      memcpy(u, new_u, N*N*sizeof(double));
      // black
      #pragma omp parallel for
      for (int j = N - 1; j > 0; j--) {
	for (int i = 1; i < N; i++) {
	  if ((i + j) % 2 == 0) {
	    float new_value = (f[i*N + j]/((N+1)*(N+1)) + u[(i-1)*N + j] + u[i*N + j-1] + u[(i+1)*N + j] + u[i*N + j+1]) / 4;
	    new_u[i*N + j] = new_value;
	  }
	}
      }
      memcpy(u, new_u, N*N*sizeof(double));
      free(new_u);
      
      count++;
    }
    free(u);
    free(f);
    double time = t.toc();
    printf("%9d %10f\n", N, time);
  }
}
