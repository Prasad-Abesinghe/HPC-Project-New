/*
 * RMSE Comparison Tool
 * Reads serial_result.bin and any parallel_result.bin and prints RMSE.
 *
 * Compile: gcc -O2 -o rmse 5_rmse.c -lm
 * Usage:   ./rmse serial_result.bin openmp_result.bin
 *          ./rmse serial_result.bin mpi_result.bin
 *          ./rmse serial_result.bin cuda_result.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <reference.bin> <test.bin>\n", argv[0]);
        return 1;
    }

    size_t total = (size_t)N * N;
    double *ref  = (double *)malloc(total * sizeof(double));
    double *test = (double *)malloc(total * sizeof(double));

    FILE *f1 = fopen(argv[1], "rb");
    FILE *f2 = fopen(argv[2], "rb");
    if (!f1 || !f2) { perror("fopen"); return 1; }

    fread(ref,  sizeof(double), total, f1);
    fread(test, sizeof(double), total, f2);
    fclose(f1); fclose(f2);

    double sum_sq = 0.0;
    double max_diff = 0.0;
    for (size_t k = 0; k < total; k++) {
        double diff = ref[k] - test[k];
        sum_sq += diff * diff;
        if (fabs(diff) > max_diff) max_diff = fabs(diff);
    }

    double rmse = sqrt(sum_sq / total);
    printf("Comparing  : %s  vs  %s\n", argv[1], argv[2]);
    printf("Grid size  : %d x %d\n", N, N);
    printf("RMSE       : %.10e\n", rmse);
    printf("Max diff   : %.10e\n", max_diff);

    if (rmse < 1e-10)
        printf("Result     : IDENTICAL (floating-point exact)\n");
    else if (rmse < 1e-6)
        printf("Result     : NUMERICALLY EQUIVALENT (within tolerance)\n");
    else
        printf("Result     : DIFFERS — check your implementation!\n");

    free(ref); free(test);
    return 0;
}
