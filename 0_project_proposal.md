# Project Proposal

**Course:** EE7218 / EC7207: High Performance Computing
**Project Title:** Parallel 2D Heat Diffusion Simulation
**Member:** Abesinghe B.M.L.P. — EG/2021/4377

---

## Project Description

Heat diffusion describes how thermal energy spreads through a material over time. In a 2D setting, this is governed by the heat equation, which is discretised using the finite difference method (Jacobi iteration) on an N × N grid. Each cell's temperature at the next time step is computed from its four neighbours, meaning every iteration performs O(N²) floating-point operations. For large grids and many iterations this becomes computationally expensive in serial, making it an ideal candidate for parallel acceleration.

This project implements a 2D heat diffusion simulation on a 512 × 512 grid with fixed boundary conditions (top edge at 100 °C, all other edges at 0 °C) for 5000 time steps. A serial baseline is first established, then the computation is parallelised using OpenMP, POSIX Threads (Pthreads), MPI, and a Hybrid (MPI + OpenMP) approach. The accuracy of each parallel version is validated against the serial result using Root Mean Square Error (RMSE).

---

## Objectives

- Implement a serial baseline for the 2D heat diffusion simulation using the Jacobi finite difference method.
- Parallelise the simulation using OpenMP and POSIX Threads for shared-memory execution.
- Parallelise the simulation using MPI with domain decomposition and ghost-row halo exchange for distributed-memory execution.
- Implement a Hybrid (MPI + OpenMP) version combining both levels of parallelism.
- Measure and compare execution time across all implementations by varying the number of threads and processes.
- Validate correctness of all parallel results against the serial baseline using RMSE analysis.

---

## Tools and Technologies

- **Language:** C
- **APIs:** OpenMP, POSIX Threads (Pthreads), MPI (Microsoft MPI / OpenMPI)
- **Accuracy Tool:** Custom RMSE utility comparing binary grid outputs
- **Build System:** GCC (MinGW-W64), Makefile
- **Environment:** Windows workstation (AMD64, multi-core)

---

## Expected Outcomes and Significance

The project is expected to demonstrate clear speedups over the serial baseline when using shared-memory parallelism (OpenMP, Pthreads) on a single workstation. MPI performance is anticipated to scale well across multiple processes when run on a distributed cluster, while showing communication overhead on a single node. The Hybrid (MPI + OpenMP) approach is expected to achieve the best balance between compute and communication by combining intra-node thread parallelism with inter-node process distribution.

This work provides practical experience in parallel algorithm design, synchronization, halo exchange communication patterns, and numerical accuracy verification in scientific computing.

---

## References

[1] Pacheco, P. S. (2011). *An Introduction to Parallel Programming*. Morgan Kaufmann.

[2] Hockney, R. W., & Eastwood, J. W. (1988). *Computer Simulation Using Particles*. Taylor & Francis.

[3] OpenMP Architecture Review Board. (2021). *OpenMP Application Programming Interface, Version 5.1*. Retrieved from https://www.openmp.org

[4] MPI Forum. (2021). *MPI: A Message-Passing Interface Standard, Version 4.0*. Retrieved from https://www.mpi-forum.org
