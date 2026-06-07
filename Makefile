# Makefile for Heat Diffusion Project
# Usage:
#   make all        — build everything (requires GCC, OpenMPI)
#   make serial     — build serial only
#   make openmp     — build OpenMP only
#   make mpi        — build MPI only
#   make hybrid     — build hybrid MPI+OpenMP
#   make rmse       — build RMSE tool
#   make clean      — remove binaries

CC      = gcc
MPICC   = mpicc
CFLAGS  = -O2 -Wall
OMP_FLAGS = -fopenmp

.PHONY: all serial openmp mpi hybrid rmse clean

all: serial openmp mpi hybrid rmse

serial:
	$(CC) $(CFLAGS) -o serial 1_serial.c -lm
	@echo "Built: serial"

openmp:
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o openmp 2_openmp.c -lm
	@echo "Built: openmp"

mpi:
	$(MPICC) $(CFLAGS) -o mpi_heat 3_mpi.c -lm
	@echo "Built: mpi_heat"

hybrid:
	$(MPICC) $(CFLAGS) $(OMP_FLAGS) -o hybrid 4_hybrid.c -lm
	@echo "Built: hybrid"

rmse:
	$(CC) $(CFLAGS) -o rmse 5_rmse.c -lm
	@echo "Built: rmse"

clean:
	rm -f serial openmp mpi_heat hybrid rmse *.bin
	@echo "Cleaned."

# --- Quick timing runs ---
run_serial:
	./serial

run_openmp_1:
	OMP_NUM_THREADS=1  ./openmp
run_openmp_2:
	OMP_NUM_THREADS=2  ./openmp
run_openmp_4:
	OMP_NUM_THREADS=4  ./openmp
run_openmp_8:
	OMP_NUM_THREADS=8  ./openmp

run_mpi_1:
	mpirun -np 1 ./mpi_heat
run_mpi_2:
	mpirun -np 2 ./mpi_heat
run_mpi_4:
	mpirun -np 4 ./mpi_heat
run_mpi_8:
	mpirun -np 8 ./mpi_heat

compare_openmp:
	./rmse serial_result.bin openmp_result.bin
compare_mpi:
	./rmse serial_result.bin mpi_result.bin
compare_hybrid:
	./rmse serial_result.bin hybrid_result.bin
