# Steps: the development history

This directory contains the intermediate programs used to build the final
project. They are kept as a readable record of the development process; the
final benchmark code is in [`../src`](../src).

All programs are independent and can be compiled with `make` in this directory.

## Step map

| Step | File | What it demonstrates |
|---|---|---|
| 1 | `hello_mpi.c` | MPI init, rank, size, processor name |
| 2 | `mpi_rows.c` | row partition formula for image bands |
| 3 | `mandelbrot_serial.c` | serial escape-time renderer and PPM writer |
| 4 | `mandelbrot_mpi.c` | MPI row bands with `MPI_Gatherv` |
| 5 | `mandelbrot_mpi_omp_static.c` | MPI + OpenMP with `schedule(static)` |
| 6 | `mandelbrot_mpi_omp_dynamic.c` | final MPI + OpenMP with `schedule(dynamic, 8)` |

The final project in `../src` uses the same code as steps 5 and 6, plus the
Makefile, run script and README.

## Build

```bash
make
```

This produces:

```text
hello_mpi
mpi_rows
mandelbrot_serial
mandelbrot_mpi
mandelbrot_mpi_omp_static
mandelbrot_mpi_omp_dynamic
```

## Run the steps

### 1. MPI hello

```bash
mpirun --bind-to none -np 4 ./hello_mpi
```

Expected: four lines with ranks 0-3. The order is not guaranteed.

### 2. Row partition

```bash
mpirun --bind-to none -np 4 ./mpi_rows 600
```

Expected:

```text
rank 0/4: y0=0 rows=150 end=150
rank 1/4: y0=150 rows=150 end=300
rank 2/4: y0=300 rows=150 end=450
rank 3/4: y0=450 rows=150 end=600
```

The ranges must cover all 600 rows without gaps or overlap.

### 3. Serial renderer

```bash
./mandelbrot_serial 400 300 500 out_serial.ppm
```

Expected hash:

```text
FNV-1a: f80f39830169dbdb
```

### 4. Pure MPI renderer

```bash
OMP_NUM_THREADS=1 mpirun --bind-to none -np 2 \
  ./mandelbrot_mpi 400 300 500 out_mpi.ppm
```

It must print the same FNV hash as the serial renderer.

### 5. Static hybrid

```bash
OMP_NUM_THREADS=4 mpirun --bind-to none \
  -x OMP_NUM_THREADS -np 1 \
  ./mandelbrot_mpi_omp_static 400 300 500 out_static.ppm
```

### 6. Dynamic hybrid

```bash
OMP_NUM_THREADS=2 mpirun --bind-to none \
  -x OMP_NUM_THREADS -np 2 \
  ./mandelbrot_mpi_omp_dynamic 400 300 500 out_dynamic.ppm
```

Steps 3-6 must all produce the same image hash. The final project uses the
800 x 600, `max_iter = 1000` configuration described in the main README.

## Relationship to the final benchmark

The step programs are smaller and are intended for learning and debugging.
The final `src/` programs add:

- three-run benchmark script;
- two-node check;
- final result tables and report;
- the same `-ffp-contract=off` build rule.

The final measurements should always be taken from `../src`, not from this
directory.
