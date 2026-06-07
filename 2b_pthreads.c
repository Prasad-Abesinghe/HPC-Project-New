/*
 * Heat Diffusion Simulation - POSIX Threads (Pthreads) Version
 *
 * Each thread owns a horizontal band of rows. A pthread_barrier
 * synchronises all threads between iterations; thread 0 applies
 * boundary conditions and swaps the two grid buffers.
 *
 * Compile: gcc -O2 -Wall -o pthreads 2b_pthreads.c -lm -lpthread
 * Run:     ./pthreads [num_threads]   (default: 4)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#define N        512
#define MAX_ITER 5000
#define ALPHA    0.25

/* Two flat grid buffers shared across all threads */
static double *g_cur, *g_next;
static pthread_barrier_t barrier;

#define G_CUR(r,c)  g_cur [(r)*N+(c)]
#define G_NEXT(r,c) g_next[(r)*N+(c)]

typedef struct {
    int id;
    int row_start;   /* first row owned (inclusive) */
    int row_end;     /* last  row owned (exclusive) */
} thread_data_t;

static void *worker(void *arg) {
    thread_data_t *td = (thread_data_t *)arg;

    for (int iter = 0; iter < MAX_ITER; iter++) {

        /* Stencil update for this thread's rows */
        for (int i = td->row_start; i < td->row_end; i++) {
            if (i == 0 || i == N - 1) continue;   /* boundary rows below */
            for (int j = 1; j < N - 1; j++) {
                G_NEXT(i,j) = G_CUR(i,j)
                    + ALPHA * (G_CUR(i+1,j) + G_CUR(i-1,j)
                             + G_CUR(i,j+1) + G_CUR(i,j-1)
                             - 4.0 * G_CUR(i,j));
            }
            G_NEXT(i, 0)     = 0.0;
            G_NEXT(i, N - 1) = 0.0;
        }

        /* All threads must finish writing g_next before thread 0 reads it */
        pthread_barrier_wait(&barrier);

        /* Thread 0: apply global boundary conditions and swap buffers */
        if (td->id == 0) {
            for (int j = 0; j < N; j++) G_NEXT(0, j)     = 100.0;
            G_NEXT(0, 0) = 0.0;          /* left column wins top-left corner */
            for (int j = 0; j < N; j++) G_NEXT(N-1, j)   =   0.0;
            double *tmp = g_cur; g_cur = g_next; g_next = tmp;
        }

        /* All threads wait for the pointer swap before next iteration */
        pthread_barrier_wait(&barrier);
    }
    return NULL;
}

int main(int argc, char **argv) {
    int nthreads = (argc > 1) ? atoi(argv[1]) : 4;
    if (nthreads < 1) nthreads = 1;

    g_cur  = (double *)calloc(N * N, sizeof(double));
    g_next = (double *)calloc(N * N, sizeof(double));

    /* Apply boundary conditions to initial grid (mirrors serial apply_boundary) */
    for (int j = 0; j < N; j++) {
        G_CUR(0,   j) = 100.0;
        G_CUR(N-1, j) =   0.0;
        G_CUR(j,   0) =   0.0;
        G_CUR(j, N-1) =   0.0;
    }

    pthread_barrier_init(&barrier, NULL, nthreads);

    pthread_t     *threads = malloc(nthreads * sizeof(pthread_t));
    thread_data_t *tdata   = malloc(nthreads * sizeof(thread_data_t));

    int rows_per_thread = N / nthreads;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int t = 0; t < nthreads; t++) {
        tdata[t].id        = t;
        tdata[t].row_start = t * rows_per_thread;
        tdata[t].row_end   = (t == nthreads - 1) ? N : tdata[t].row_start + rows_per_thread;
        pthread_create(&threads[t], NULL, worker, &tdata[t]);
    }
    for (int t = 0; t < nthreads; t++)
        pthread_join(threads[t], NULL);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    printf("Pthreads Heat Diffusion (%dx%d, %d iters, %d threads)\n",
           N, N, MAX_ITER, nthreads);
    printf("Center temperature : %.6f deg C\n", G_CUR(N/2, N/2));
    printf("Elapsed time       : %.4f seconds\n", elapsed);

    FILE *fp = fopen("pthreads_result.bin", "wb");
    fwrite(g_cur, sizeof(double), N * N, fp);
    fclose(fp);
    printf("Result saved to pthreads_result.bin\n");

    pthread_barrier_destroy(&barrier);
    free(threads); free(tdata); free(g_cur); free(g_next);
    return 0;
}
