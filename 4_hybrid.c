/*
 * Heat Diffusion Simulation - Hybrid MPI + OpenMP Version
 *
 * Two-level parallelism:
 *   MPI  — splits rows into slabs across processes (distributed memory)
 *   OpenMP — parallelises the stencil loop within each slab (shared memory)
 *
 * Compile: mpicc -O2 -Wall -fopenmp -o hybrid 4_hybrid.c -lm
 * Run:     mpiexec -np 2 cmd /c "set OMP_NUM_THREADS=4 && ./hybrid.exe"
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define N        512
#define MAX_ITER 5000
#define ALPHA    0.25

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = N / size;
    int row_start = rank * rows_per_proc;
    int row_end   = (rank == size - 1) ? N : row_start + rows_per_proc;
    int local_rows = row_end - row_start;
    int total_rows = local_rows + 2;   /* +2 ghost rows */

    double *cur  = (double *)calloc(total_rows * N, sizeof(double));
    double *next = (double *)calloc(total_rows * N, sizeof(double));

    #define CUR(r,c)  cur [(r)*N+(c)]
    #define NEXT(r,c) next[(r)*N+(c)]

    /* Initialise boundary conditions on owned rows */
    for (int i = 1; i <= local_rows; i++) {
        int global_row = row_start + (i - 1);
        for (int j = 0; j < N; j++)
            CUR(i,j) = (global_row == 0) ? 100.0 : 0.0;
        CUR(i, 0)     = 0.0;
        CUR(i, N - 1) = 0.0;
    }

    int nthreads = omp_get_max_threads();
    double t0 = MPI_Wtime();

    for (int iter = 0; iter < MAX_ITER; iter++) {

        /* --- Halo exchange (non-blocking) --- */
        MPI_Request reqs[4];
        int nreq = 0;

        if (rank < size - 1) {
            MPI_Isend(&CUR(local_rows, 0), N, MPI_DOUBLE,
                      rank + 1, 0, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Irecv(&CUR(local_rows + 1, 0), N, MPI_DOUBLE,
                      rank + 1, 1, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (rank > 0) {
            MPI_Isend(&CUR(1, 0), N, MPI_DOUBLE,
                      rank - 1, 1, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Irecv(&CUR(0, 0), N, MPI_DOUBLE,
                      rank - 1, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        /* --- Stencil update — parallelised with OpenMP --- */
        #pragma omp parallel for schedule(static)
        for (int i = 1; i <= local_rows; i++) {
            int global_row = row_start + (i - 1);
            if (global_row == 0 || global_row == N - 1) continue;

            for (int j = 1; j < N - 1; j++) {
                NEXT(i,j) = CUR(i,j)
                    + ALPHA * (CUR(i+1,j) + CUR(i-1,j)
                             + CUR(i,j+1) + CUR(i,j-1)
                             - 4.0 * CUR(i,j));
            }
            NEXT(i, 0)     = 0.0;
            NEXT(i, N - 1) = 0.0;
        }

        /* Global boundary rows */
        if (row_start == 0) {
            for (int j = 0; j < N; j++) NEXT(1, j) = 100.0;
            NEXT(1, 0) = 0.0;   /* left column wins top-left corner */
        }
        if (row_end == N)
            for (int j = 0; j < N; j++) NEXT(local_rows, j) = 0.0;

        double *tmp = cur; cur = next; next = tmp;
    }

    double elapsed = MPI_Wtime() - t0;

    /* Gather centre temperature */
    double center_temp = 0.0;
    int center_rank = (N / 2) / rows_per_proc;
    if (rank == center_rank) {
        int local_center_row = (N / 2) - row_start + 1;
        center_temp = CUR(local_center_row, N / 2);
        if (center_rank != 0)
            MPI_Send(&center_temp, 1, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        if (center_rank != 0)
            MPI_Recv(&center_temp, 1, MPI_DOUBLE,
                     center_rank, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Hybrid MPI+OpenMP Heat Diffusion (%dx%d, %d iters, %d procs x %d threads)\n",
               N, N, MAX_ITER, size, nthreads);
        printf("Center temperature : %.6f degC\n", center_temp);
        printf("Elapsed time       : %.4f seconds\n", elapsed);
    }

    /* Gather full grid to rank 0 and save */
    double *full_grid = NULL;
    if (rank == 0) full_grid = (double *)malloc(N * N * sizeof(double));

    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0) {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs     = (int *)malloc(size * sizeof(int));
        for (int r = 0; r < size; r++) {
            int rs = r * rows_per_proc;
            int re = (r == size - 1) ? N : rs + rows_per_proc;
            sendcounts[r] = (re - rs) * N;
            displs[r]     = rs * N;
        }
    }

    double *send_buf = (double *)malloc(local_rows * N * sizeof(double));
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < N; j++)
            send_buf[i * N + j] = CUR(i + 1, j);

    MPI_Gatherv(send_buf, local_rows * N, MPI_DOUBLE,
                full_grid, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE *fp = fopen("hybrid_result.bin", "wb");
        fwrite(full_grid, sizeof(double), N * N, fp);
        fclose(fp);
        printf("Result saved to hybrid_result.bin\n");
        free(full_grid); free(sendcounts); free(displs);
    }

    free(send_buf); free(cur); free(next);
    MPI_Finalize();
    return 0;
}
