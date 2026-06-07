/*
 * Heat Diffusion Simulation - Serial Version
 * 2D plate with fixed boundary conditions.
 * Uses finite difference method (Jacobi iteration).
 *
 * Compile: gcc -O2 -o serial 1_serial.c -lm
 * Run:     ./serial
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N        512       /* Grid size N x N */
#define MAX_ITER 5000      /* Number of time steps */
#define ALPHA    0.25      /* Diffusion coefficient (must be <= 0.25 for stability) */

/* Allocate a 2D grid */
double **alloc_grid(int n) {
    double **g = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++)
        g[i] = (double *)calloc(n, sizeof(double));
    return g;
}

/* Free a 2D grid */
void free_grid(double **g, int n) {
    for (int i = 0; i < n; i++) free(g[i]);
    free(g);
}

/* Apply boundary conditions:
 *   Top edge    = 100°C (heat source)
 *   Other edges = 0°C   (cold boundary)
 */
void apply_boundary(double **g, int n) {
    for (int j = 0; j < n; j++) {
        g[0][j]     = 100.0;   /* top */
        g[n-1][j]   =   0.0;   /* bottom */
        g[j][0]     =   0.0;   /* left */
        g[j][n-1]   =   0.0;   /* right */
    }
}

int main(void) {
    double **cur  = alloc_grid(N);
    double **next = alloc_grid(N);

    apply_boundary(cur,  N);
    apply_boundary(next, N);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        /* Update interior cells */
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                next[i][j] = cur[i][j]
                    + ALPHA * (cur[i+1][j] + cur[i-1][j]
                             + cur[i][j+1] + cur[i][j-1]
                             - 4.0 * cur[i][j]);
            }
        }
        apply_boundary(next, N);

        /* Swap grids */
        double **tmp = cur;
        cur  = next;
        next = tmp;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec)
                   + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    /* Print center temperature as a sanity check */
    printf("Serial Heat Diffusion (%dx%d, %d iterations)\n", N, N, MAX_ITER);
    printf("Center temperature : %.6f °C\n", cur[N/2][N/2]);
    printf("Elapsed time       : %.4f seconds\n", elapsed);

    /* Save final grid to file for RMSE comparison */
    FILE *fp = fopen("serial_result.bin", "wb");
    for (int i = 0; i < N; i++)
        fwrite(cur[i], sizeof(double), N, fp);
    fclose(fp);
    printf("Result saved to serial_result.bin\n");

    free_grid(cur,  N);
    free_grid(next, N);
    return 0;
}
