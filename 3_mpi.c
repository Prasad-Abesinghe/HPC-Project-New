/*
 * Heat Diffusion Simulation - Distributed Memory (MPI)
 *
 * Domain decomposition: the grid is split into horizontal SLABS,
 * one slab per MPI process. Ghost rows are exchanged each iteration.
 *
 * Compile: mpicc -O2 -o mpi 3_mpi.c -lm
 * Run:     mpirun -np 4 ./mpi
 *          (change 4 to 1, 2, 4, 8 for timing comparison)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N        512
#define MAX_ITER 5000
#define ALPHA    0.25

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---------------------------------------------------------
     * Divide rows among processes.
     * Each process owns rows [row_start, row_end).
     * Two extra ghost rows (top & bottom) are added for halo exchange.
     * --------------------------------------------------------- */
    int rows_per_proc = N / size;
    int row_start = rank * rows_per_proc;
    int row_end   = (rank == size - 1) ? N : row_start + rows_per_proc;
    int local_rows = row_end - row_start;   /* actual owned rows */
    int total_rows = local_rows + 2;        /* +2 ghost rows     */

    /* Allocate local slab (total_rows × N) as flat 1-D array */
    double *cur  = (double *)calloc(total_rows * N, sizeof(double));
    double *next = (double *)calloc(total_rows * N, sizeof(double));

    /* Convenience macro: row r (0-based within total_rows), col c */
    #define CUR(r,c)  cur [(r)*N+(c)]
    #define NEXT(r,c) next[(r)*N+(c)]

    /* Initialise boundary conditions on owned rows */
    for (int i = 1; i <= local_rows; i++) {
        int global_row = row_start + (i - 1);
        for (int j = 0; j < N; j++) {
            if (global_row == 0)     CUR(i,j) = 100.0;  /* top edge */
            else                     CUR(i,j) =   0.0;
        }
        CUR(i, 0)     = 0.0;   /* left  */
        CUR(i, N - 1) = 0.0;   /* right */
    }

    double t0 = MPI_Wtime();

    for (int iter = 0; iter < MAX_ITER; iter++) {

        /* --- Halo exchange (non-blocking for overlap) --- */
        MPI_Request reqs[4];
        int nreq = 0;

        /* Send bottom owned row DOWN; receive from below into bottom ghost */
        if (rank < size - 1) {
            MPI_Isend(&CUR(local_rows, 0), N, MPI_DOUBLE,
                      rank + 1, 0, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Irecv(&CUR(local_rows + 1, 0), N, MPI_DOUBLE,
                      rank + 1, 1, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        /* Send top owned row UP; receive from above into top ghost */
        if (rank > 0) {
            MPI_Isend(&CUR(1, 0), N, MPI_DOUBLE,
                      rank - 1, 1, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Irecv(&CUR(0, 0), N, MPI_DOUBLE,
                      rank - 1, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        /* --- Update interior cells --- */
        for (int i = 1; i <= local_rows; i++) {
            int global_row = row_start + (i - 1);
            /* Skip true boundary rows (handled by BC) */
            if (global_row == 0 || global_row == N - 1) continue;

            for (int j = 1; j < N - 1; j++) {
                NEXT(i,j) = CUR(i,j)
                    + ALPHA * (CUR(i+1,j) + CUR(i-1,j)
                             + CUR(i,j+1) + CUR(i,j-1)
                             - 4.0 * CUR(i,j));
            }
            /* Left/right boundaries */
            NEXT(i, 0)     = 0.0;
            NEXT(i, N - 1) = 0.0;
        }
        /* Top/bottom global boundary rows */
        if (row_start == 0) {
            for (int j = 0; j < N; j++) NEXT(1, j) = 100.0;
            NEXT(1, 0) = 0.0;   /* left column wins top-left corner, matching serial */
        }
        if (row_end == N)
            for (int j = 0; j < N; j++) NEXT(local_rows, j) = 0.0;

        /* Swap */
        double *tmp = cur; cur = next; next = tmp;
    }

    double elapsed = MPI_Wtime() - t0;

    /* Gather centre temperature from owning process */
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
        printf("MPI Heat Diffusion (%dx%d, %d iters, %d procs)\n",
               N, N, MAX_ITER, size);
        printf("Center temperature : %.6f °C\n", center_temp);
        printf("Elapsed time       : %.4f seconds\n", elapsed);
    }

    /* Gather full grid to rank 0 and save */
    double *full_grid = NULL;
    if (rank == 0) full_grid = (double *)malloc(N * N * sizeof(double));

    /* Each process sends its owned rows (skip ghost rows) */
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
    /* Build contiguous send buffer (owned rows only, no ghosts) */
    double *send_buf = (double *)malloc(local_rows * N * sizeof(double));
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < N; j++)
            send_buf[i * N + j] = CUR(i + 1, j);  /* skip top ghost (row 0) */

    MPI_Gatherv(send_buf, local_rows * N, MPI_DOUBLE,
                full_grid, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE *fp = fopen("mpi_result.bin", "wb");
        fwrite(full_grid, sizeof(double), N * N, fp);
        fclose(fp);
        printf("Result saved to mpi_result.bin\n");
        free(full_grid); free(sendcounts); free(displs);
    }

    free(send_buf); free(cur); free(next);
    MPI_Finalize();
    return 0;
}
