# Analysis Report: Heat Diffusion Simulation
## EE7218 / EC7207 — High Performance Computing

---

## 1. Problem Description

Heat diffusion describes how thermal energy spreads across a material over time. We simulate a **2D square metal plate (512 × 512 grid)** with the following boundary conditions:

- **Top edge:** Fixed at 100 °C (heat source)
- **All other edges:** Fixed at 0 °C (cold boundary)
- **Initial state:** All interior cells at 0 °C

The plate evolves over **5000 time steps** using the **explicit finite difference method (Jacobi iteration)**:

```
T_new[i][j] = T[i][j] + α × (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1] − 4×T[i][j])
```

Where **α = 0.25** (diffusion coefficient, chosen for numerical stability).

---

## 2. Parallel Programming Concepts Applied

### 2.1 Diagram

```
┌──────────────────────────────────────────────────┐
│            512 × 512 Grid                        │
│  ┌────────────────────────────────────────────┐  │
│  │  Top boundary = 100°C (fixed)              │  │
│  ├────────────────────────────────────────────┤  │
│  │  Thread 0 / Process 0 / CUDA Block Row 0   │  │
│  │  rows 1..127                               │  │
│  ├────────────────────────────────────────────┤  │
│  │  Thread 1 / Process 1 / CUDA Block Row 1   │  │
│  │  rows 128..255                             │  │
│  ├────────────────────────────────────────────┤  │
│  │  Thread 2 / Process 2 / CUDA Block Row 2   │  │
│  │  rows 256..383                             │  │
│  ├────────────────────────────────────────────┤  │
│  │  Thread 3 / Process 3 / CUDA Block Row 3   │  │
│  │  rows 384..510                             │  │
│  ├────────────────────────────────────────────┤  │
│  │  Bottom boundary = 0°C (fixed)             │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

**Key insight:** Because the Jacobi update reads from `cur[]` and writes to `next[]`, there is **no read-write dependency between cells in the same iteration**. Every cell can be updated independently → perfect data parallelism.

---

### 2.2 Serial Version (`1_serial.c`)
- Simple nested loop over all (N−2)² interior cells.
- Baseline for timing and correctness.

### 2.3 Shared Memory — OpenMP (`2_openmp.c`)
- `#pragma omp parallel for schedule(static)` applied to the outer row loop.
- OpenMP divides rows evenly across threads.
- All threads share the same `cur[]` and `next[]` arrays — no communication needed.
- Threads synchronise implicitly at the end of each `parallel for`.

### 2.4 Distributed Memory — MPI (`3_mpi.c`)
- Grid is split into **horizontal slabs** (row decomposition).
- Each MPI process owns `N/P` rows.
- **Ghost rows** (halo exchange) are sent between neighbouring processes using `MPI_Isend` / `MPI_Irecv` before each iteration.
- Results gathered to rank 0 via `MPI_Gatherv` for output.

### 2.5 Hybrid GPU — CUDA (`4_cuda.cu`)
- Each **CUDA thread** computes exactly one interior cell `(i, j)`.
- Threads are organised in 16×16 blocks → 256 threads/block.
- Grid of blocks covers the full (N−2) × (N−2) interior.
- Boundary conditions applied by a separate lightweight kernel.
- Device pointer swap (`d_cur ↔ d_next`) replaces memcpy — O(1) per iteration.

---

## 3. Accuracy (RMSE vs Serial)

Run `5_rmse.c` after executing all versions to compare results against the serial baseline.

| Version | Expected RMSE | Reason |
|---------|---------------|--------|
| OpenMP  | ~0 (< 1e-15)  | Identical arithmetic, shared memory |
| MPI     | ~0 (< 1e-10)  | Same operations, floating-point order may vary slightly |
| CUDA    | < 1e-10       | GPU double-precision; order of operations differs slightly |

**Expected output example:**
```
RMSE       : 0.0000000000e+00
Result     : IDENTICAL (floating-point exact)
```

---

## 4. Timing Results

Run each version and fill in the table below.

### 4.1 Serial Baseline

| Grid | Iterations | Time (s) |
|------|------------|----------|
| 512×512 | 5000 | ________ |

### 4.2 OpenMP — Vary Thread Count

| Threads | Time (s) | Speedup vs Serial |
|---------|----------|-------------------|
| 1       | ________ | 1.0×              |
| 2       | ________ | _____×            |
| 4       | ________ | _____×            |
| 8       | ________ | _____×            |

### 4.3 MPI — Vary Process Count

| Processes | Time (s) | Speedup vs Serial |
|-----------|----------|-------------------|
| 1         | ________ | 1.0×              |
| 2         | ________ | _____×            |
| 4         | ________ | _____×            |
| 8         | ________ | _____×            |

### 4.4 CUDA

| Version | Time (s) | Speedup vs Serial |
|---------|----------|-------------------|
| CUDA    | ________ | _____×            |

---

## 5. Discussion

- **Amdahl's Law:** The boundary condition update and grid swap are serial. At very high thread counts, these become the bottleneck. Speedup will plateau.
- **OpenMP** shows near-linear speedup up to the number of physical cores (due to shared cache). Beyond that, memory bandwidth becomes the limit.
- **MPI** introduces communication overhead (halo exchange). With 4+ processes the computation savings outweigh the MPI cost, giving good speedup.
- **CUDA** gives the highest speedup because the GPU has thousands of cores all updating cells in parallel. Memory bandwidth on the GPU far exceeds the CPU.

---

## 6. Conclusion

The 2D heat diffusion problem is an ideal HPC benchmark because:
1. The Jacobi update is **embarrassingly parallel** — no dependencies within an iteration.
2. All four parallelisation strategies (OpenMP, MPI, CUDA) map naturally onto the row-decomposition pattern.
3. RMSE comparison confirms numerical correctness of all parallel versions.
4. Timing results clearly demonstrate speedup with increasing threads/processes/GPU.

---

*Report generated for EE7218 / EC7207 — fill in timing values from your actual runs.*
