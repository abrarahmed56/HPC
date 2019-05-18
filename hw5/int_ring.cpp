#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
  int rank, size;
  MPI_Init( &argc, &argv );
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  int i = 0;
  int num_repeats = 100;
  int* msg = (int*) malloc(1e6);
  //int msg = 0;
  double time_taken = MPI_Wtime();
  MPI_Status status;
  for (i=0; i < num_repeats; i++) {
    if (rank == size - 1) {
      MPI_Recv(msg, size, MPI_INT, rank-1, i, MPI_COMM_WORLD, &status);
      //printf("recieved message %d from rank %d\n", msg, rank-1);
      //msg += rank;
      MPI_Send(msg, size, MPI_INT, 0, i, MPI_COMM_WORLD);
      //printf("send message %d to rank 0\n", msg);
    }
    else {
      if (rank != 0) {
	MPI_Recv(msg, size, MPI_INT, rank-1, i, MPI_COMM_WORLD, &status);
	//printf("recieved message %d from rank %d\n", msg, rank-1);
      }
      //msg += rank;
      MPI_Send(msg, size, MPI_INT, rank+1, i, MPI_COMM_WORLD);
      //printf("send message %d to rank 0\n", msg);
      if (rank == 0) {
	MPI_Recv(msg, size, MPI_INT, size-1, i, MPI_COMM_WORLD, &status);
	//printf("recieved message %d from rank %d\n", msg, size-1);
      }
      //MPI_SEND(rank+1, 1, MPI_INT);
      }
  //printf("I am %d of %d!\n", rank, size);
  }
  time_taken = MPI_Wtime() - time_taken;
  if (rank == 0) {
    //printf("Result: %d for num_repeats=%d\n", msg, num_repeats);
    printf("Latency: %f ms\n", time_taken/num_repeats * 1000);
    // multiply by 1e6 for the number of bytes transmitted, divide by 1e9 for GB
    double bandwidth = size*num_repeats/time_taken*(1e6)/(1e9);
    printf("Bandwidth: %f GB/s\n", bandwidth);
    }
  MPI_Finalize();

  return 0;
}

/*
#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

double time_pingpong(int proc0, int proc1, long Nrepeat, long Nsize, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  char* msg = (char*) malloc(Nsize);
  for (long i = 0; i < Nsize; i++) msg[i] = 42;

  MPI_Barrier(comm);
  double tt = MPI_Wtime();
  for (long repeat  = 0; repeat < Nrepeat; repeat++) {
    MPI_Status status;
    if (repeat % 1 == 0) { // even iterations

      if (rank == proc0)
        MPI_Send(msg, Nsize, MPI_CHAR, proc1, repeat, comm);
      else if (rank == proc1)
        MPI_Recv(msg, Nsize, MPI_CHAR, proc0, repeat, comm, &status);

    }
    else { // odd iterations

      if (rank == proc0)
        MPI_Recv(msg, Nsize, MPI_CHAR, proc0, repeat, comm, &status);
      else if (rank == proc1)
        MPI_Send(msg, Nsize, MPI_CHAR, proc1, repeat, comm);

    }
  }
  tt = MPI_Wtime() - tt;

  free(msg);
  return tt;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  if (argc < 3) {
    printf("Usage: mpirun ./pingpong <process-rank0> <process-rank1>\n");
    abort();
  }
  int proc0 = atoi(argv[1]);
  int proc1 = atoi(argv[2]);

  int rank;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);

  long Nrepeat = 1000;
  double tt = time_pingpong(proc0, proc1, Nrepeat, 1, comm);
  if (!rank) printf("pingpong latency: %e ms\n", tt/Nrepeat * 1000);

  Nrepeat = 10000;
  long Nsize = 1000000;
  tt = time_pingpong(proc0, proc1, Nrepeat, Nsize, comm);
  if (!rank) printf("pingpong bandwidth: %e GB/s\n", (Nsize*Nrepeat)/tt/1e9);

  MPI_Finalize();
}
*/
