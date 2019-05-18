/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/* compute global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq){
  int i;
  double tmp, gres = 0.0, lres = 0.0;

  for (i = 1; i <= lN; i++){
    tmp = ((2.0*lu[i] - lu[i-1] - lu[i+1]) * invhsq - 1);
    lres += tmp * tmp;
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]){
  int mpirank, i, p, N, lN, iter, max_iters;
  MPI_Status status_top, status_bottom, status_left, status_right;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);

  /* compute number of unknowns handled by each process */
  lN = N / p;
  int num_processes_per_row = sqrt(p);
  if ((N % p != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (lN + 2)*(lN + 2));
  double * lunew = (double *) calloc(sizeof(double), (lN + 2)*(lN + 2));
  double * lutemp;

  double* top_points_to_send = lu;
  double* bottom_points_to_send = lu+((lN-1)*lN);
  double* left_points_to_send = (double*) calloc(sizeof(double), lN);
  double* right_points_to_send = (double*) calloc(sizeof(double), lN);
  for (int i = 0; i < lN; i++) {
    *(left_points_to_send+i) = *(lu+i*lN);
    *(right_points_to_send+i) = *(lu+((i+1)*lN)-1);
  }

  double* top_points_to_receive = (double*) calloc(sizeof(double), lN);
  double* bottom_points_to_receive = (double*) calloc(sizeof(double), lN);
  double* left_points_to_receive = (double*) calloc(sizeof(double), lN);
  double* right_points_to_receive = (double*) calloc(sizeof(double), lN);


  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

    /* Jacobi step for local points */
    for (i = 1; i <= lN; i++){
      lunew[i]  = 0.5 * (hsq + lu[i - 1] + lu[i + 1]);
    }

    /* communicate ghost values */
    int top_rank = mpirank + num_processes_per_row;
    int bottom_rank = mpirank - num_processes_per_row;
    int left_rank = mpirank - 1;
    int right_rank = mpirank + 1;

    int TOP_TO_BOTTOM = 1;
    int BOTTOM_TO_TOP = 2;
    int LEFT_TO_RIGHT = 3;
    int RIGHT_TO_LEFT = 4;

    // communicate with top if applicable
    if (top_rank < p) {
      MPI_Send(top_points_to_send, lN, MPI_DOUBLE, top_rank, TOP_TO_BOTTOM, MPI_COMM_WORLD);
      MPI_Recv(top_points_to_receive, lN, MPI_DOUBLE, top_rank, BOTTOM_TO_TOP, MPI_COMM_WORLD, &status_top);
    }
    // communicate with bottom if applicable
    if (bottom_rank >= 0) {
      MPI_Send(bottom_points_to_send, lN, MPI_DOUBLE, bottom_rank, BOTTOM_TO_TOP, MPI_COMM_WORLD);
      MPI_Recv(bottom_points_to_receive, lN, MPI_DOUBLE, bottom_rank, TOP_TO_BOTTOM, MPI_COMM_WORLD, &status_bottom);
    }
    // communicate with left if applicable
    if (mpirank % num_processes_per_row != 0 ) {
      MPI_Send(left_points_to_send, lN, MPI_DOUBLE, left_rank, LEFT_TO_RIGHT, MPI_COMM_WORLD);
      MPI_Recv(left_points_to_receive, lN, MPI_DOUBLE, left_rank, RIGHT_TO_LEFT, MPI_COMM_WORLD, &status_left);
    }
    // communicate with right if applicable
    if (mpirank % num_processes_per_row != num_processes_per_row - 1) {
      MPI_Send(right_points_to_send, lN, MPI_DOUBLE, right_rank, RIGHT_TO_LEFT, MPI_COMM_WORLD);
      MPI_Recv(right_points_to_receive, lN, MPI_DOUBLE, right_rank, LEFT_TO_RIGHT, MPI_COMM_WORLD, &status_right);
    }

    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
