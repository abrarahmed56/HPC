// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 10;

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  /*
  printf("on process %d, initial points: ", rank);
  for (int i = 0; i < N; i++) {
    printf("%d, ", vec[i]);
  }
  printf("\n");
  */
  //printf("rank: %d, first entry: %d\n", rank, vec[0]);

  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  // sort locally
  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int* splitters = (int*) malloc((p-1)*sizeof(int));
  //printf("splitters: ");
  for (int i = 0; i < p-1; i++) {
    splitters[i] = vec[(N/p)*(i+1)];
    //printf("%d, ", splitters[i]);
  }
  //printf("\n");

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  int* all_splitters = (int*) malloc(p*(p-1)*sizeof(int));
  MPI_Gather(splitters, p-1, MPI_INT, all_splitters, p-1, MPI_INT, 0, MPI_COMM_WORLD);

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  int* new_splitters = (int*) malloc((p-1)*sizeof(int));
  if (rank == 0) {
    std::sort(all_splitters, all_splitters+(p*(p-1)));
    /*
    printf("Process root has splitters:\n");
    for (int i = 0; i < (p*(p-1)); i++) {
      printf("%d, ", all_splitters[i]);
    }
    printf("\n");
    */
    //printf("new splitters: ");
    for (int i = 0; i < p-1; i++) {
      new_splitters[i] = all_splitters[(p-1)*(i+1)];
      //printf("%d, ", new_splitters[i]);
    }
    //printf("\n");
  }

  // root process broadcasts splitters to all other processes
  MPI_Bcast(new_splitters, p-1, MPI_INT, 0, MPI_COMM_WORLD);

  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;

  //std::vector<int> splitters_vec(new_splitters, new_splitters+p-1);
  int* send_counts = (int*) calloc(p, sizeof(int));
  int* receive_counts = (int*) calloc(p, sizeof(int));
  int* send_displacements = (int*) calloc(p, sizeof(int));
  int* receive_displacements = (int*) calloc(p, sizeof(int));

  //printf("vector of process %d: ", rank);
  for (int i = 0; i < N; i++) {
    int val = vec[i];
    long lower = std::lower_bound(new_splitters, new_splitters+p-1, val) - new_splitters;
    send_counts[lower] += 1;
    //printf("%d, ", val);
    //printf("[p %d] %d should go in bin %d\n", rank, vec[i], lower);
  }
  //printf("\n");

  for (int i = 0; i < p; i++) {
    //send_displacements[i+1] = std::lower_bound(vec, vec+N, splitters[i]) - vec;
    send_displacements[i+1] = send_displacements[i] + send_counts[i];
  }

  /*
  printf("on process %d, send counts: ", rank);
  for (int i = 0; i < p; i++) {
    printf("%d, ", send_counts[i]);
  }
  printf("\n");
  */
  /*
  printf("on process %d, send displacements: ", rank);
  for (int i = 0; i < p; i++) {
    printf("%d, ", send_displacements[i]);
  }
  printf("\n");
  */

  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data
  MPI_Alltoall(send_counts, 1, MPI_INT, receive_counts, 1, MPI_INT, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  int points_to_process = 0;

  //printf("on process %d, receive counts: ", rank);
  for (int i = 0; i < p; i++) {
    //printf("%d, ", receive_counts[i]);
    points_to_process += receive_counts[i];
    receive_displacements[i+1] = receive_displacements[i] + receive_counts[i];
  }
  //printf("\n");

  //printf("process %d is supposed to take %d points\n", rank, points_to_process);

  int* new_vec = (int*)malloc(points_to_process*sizeof(int));
  MPI_Alltoallv(vec, send_counts, send_displacements, MPI_INT, new_vec, receive_counts, receive_displacements, MPI_INT, MPI_COMM_WORLD);

  // do a local sort of the received data
  std::sort(new_vec, new_vec+points_to_process);

  MPI_Barrier(MPI_COMM_WORLD);
  tt = MPI_Wtime() - tt;

  // every process writes its result to a file
  FILE* fd = NULL;
  char filename[256];
  snprintf(filename, 256, "sorted_by_process%d.txt", rank);
  fd = fopen(filename, "w");

  if (fd == NULL) {
    printf("Error opening file\n");
    return 1;
  }

  for (int i = 0; i < points_to_process-1; i++) {
    //printf("%d, ", new_vec[i]);
    fprintf(fd, "%d, ", new_vec[i]);
  }
  fprintf(fd, "%d", new_vec[points_to_process-1]);
  fclose(fd);

  if (rank == 0) {
    printf("Program completed in %f s\n", tt);
  }

  free(vec);
  free(splitters);
  free(all_splitters);
  free(new_splitters);
  free(send_counts);
  free(receive_counts);
  free(send_displacements);
  free(receive_displacements);
  free(new_vec);
  MPI_Finalize();
  return 0;
}
