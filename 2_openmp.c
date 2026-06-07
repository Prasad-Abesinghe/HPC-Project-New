/*
 * Heat Diffusion Simulation - Shared Memory (OpenMP)
 *
 * Compile: gcc -O2 -fopenmp -o openmp 2_openmp.c -lm
 * Run:     OMP_NUM_THREADS=4 ./openmp
 *          (change 4 to 1, 2, 4, 8 for timing comparison)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N        512
#define MAX_ITER 5000
#define ALPHA    0.25

double **alloc_grid(int n) {
    double **g = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++)
        g[i] = (double *)calloc(n, sizeof(double));
    return g;
}

void free_grid(double **g, int n) {
    for (int i = 0; i < n; i++) free(g[i]);
    free(g);
}

void apply_boundary(double **g, int n) {
    for (int j = 0; j < n; j++) {
        g[0][j]   = 100.0;
        g[n-1][j] =   0.0;
        g[j][0]   =   0.0;
        g[j][n-1] =   0.0;
    }
}

int main(void) {
    int num_threads = omp_get_max_threads();

    double **cur  = alloc_grid(N);
    double **next = alloc_grid(N);
    apply_boundary(cur,  N);
    apply_boundary(next, N);

    double t0 = omp_get_wtime();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        /*
         * Parallelise the two nested loops over interior cells.
         * Each thread independently computes a contiguous block of rows.
         * No data dependency exists between cells within the same iteration
         * (Jacobi update reads cur[], writes next[]) — so no race condition.
         */
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                next[i][j] = cur[i][j]
                    + ALPHA * (cur[i+1][j] + cur[i-1][j]
                             + cur[i][j+1] + cur[i][j-1]
                             - 4.0 * cur[i][j]);
            }
        }

        /* Boundary is fast; no need to parallelise */
        apply_boundary(next, N);

        double **tmp = cur;
        cur  = next;
        next = tmp;
    }

    double elapsed = omp_get_wtime() - t0;

    printf("OpenMP Heat Diffusion (%dx%d, %d iters, %d threads)\n",
           N, N, MAX_ITER, num_threads);
    printf("Center temperature : %.6f °C\n", cur[N/2][N/2]);
    printf("Elapsed time       : %.4f seconds\n", elapsed);

    /* Save result for RMSE comparison against serial */
    FILE *fp = fopen("openmp_result.bin", "wb");
    for (int i = 0; i < N; i++)
        fwrite(cur[i], sizeof(double), N, fp);
    fclose(fp);
    printf("Result saved to openmp_result.bin\n");

    free_grid(cur,  N);
    free_grid(next, N);
    return 0;
}
