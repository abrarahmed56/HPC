#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  int num_threads;
  long num_to_process;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();
    num_to_process = n / num_threads;
    long start_index = thread_id * num_to_process;
    if (thread_id == 0) {
      prefix_sum[start_index] = 0;
    }
    else {
      prefix_sum[start_index] = A[start_index-1];
    }
    for (long i = start_index+1; i < num_to_process+start_index; i++) {
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    }
  }
  for (int i = 1; i < num_threads; i++) {
    long amount_to_add = prefix_sum[(num_to_process*i)-1];
    #pragma omp parallel for
    for (long j = num_to_process*i; j < num_to_process*(i+1); j++) {
      prefix_sum[j] = prefix_sum[j] + amount_to_add;
    }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
