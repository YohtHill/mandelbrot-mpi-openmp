# Mandelbrot: Hybrid MPI + OpenMP Renderer

[English](README.md) | [简体中文](README.zh-CN.md)

A hybrid MPI + OpenMP renderer for the Lab 9 final parallel programming
challenge. The image is divided into row bands across MPI ranks; each rank
renders its band with OpenMP threads; rank 0 assembles the bands with
`MPI_Gatherv` and writes a binary PPM image.

The project compares static and dynamic OpenMP scheduling, verifies that all
configurations produce exactly the same image, and explains the measured
speed-up using the non-uniform cost of Mandelbrot rows.

![Mandelbrot render](docs/mandelbrot.png)

## 1. What the program does

For every pixel, the renderer:

1. maps the pixel centre to a complex number `c = cr + i*ci`;
2. starts with `z = 0`;
3. repeats `z = z*z + c` until `|z| > 2` or `max_iter` is reached;
4. colours the pixel by the number of iterations it survived;
5. writes the RGB bytes to a binary PPM file;
6. computes an FNV-1a hash of the complete image.

The measured image is **800 x 600** with **max_iter = 1000**. The final PPM is
1,440,015 bytes: 1,440,000 RGB bytes plus a 15-byte `P6` header.

The program is deterministic. Every configuration must print:

```text
FNV-1a: 92193b63bbd42a23
```

and every PPM must have the same SHA-256:

```text
4606e1e408496266fd928852c8b0c4137113773c7d01f4b814b3ad058827c17e
```

## 2. Mandelbrot details

### 2.1 Complex plane window

The image uses:

```text
xmin = -2.10
xmax =  0.70
xspan = xmax - xmin
yspan = xspan * height / width
ymin = -0.5 * yspan
ymax =  0.5 * yspan
```

The vertical span is derived from the horizontal span so that pixels remain
square. The centre of the image is the origin of the complex plane.

### 2.2 Pixel to complex number

Pixel centres, not corners, are sampled:

```text
cr = xmin + (xmax - xmin) * (px + 0.5) / width
ci = ymax - (ymax - ymin) * (py + 0.5) / height
```

### 2.3 Escape-time loop

```c
zr = 0.0;
zi = 0.0;
iter = 0;

while (iter < max_iter && zr*zr + zi*zi <= 4.0) {
    next_r = zr*zr - zi*zi + cr;
    next_i = 2.0*zr*zi + ci;
    zr = next_r;
    zi = next_i;
    iter++;
}
```

A pixel that reaches `max_iter` is treated as inside the set and coloured
black. An escaping pixel is coloured from its iteration count.

### 2.4 Colour palette

For an escaping pixel:

```text
t = sqrt(iter / 200.0)          // clamp to 1.0
R = 255 * min(1, 2t)
G = 255 * min(1, 2t - 0.5)
B = 255 * min(1, 1 - 2t)
```

The square root stretches the low iteration counts, so the outer region is not
a single flat colour. The palette has no effect on parallel behaviour; it only
makes the expensive boundary region visible.

## 3. Serial reference

`src/mandelbrot_serial.c` is the reference implementation. It renders the whole
image in one process, writes the PPM, and prints the elapsed time and the FNV
hash.

Measured serial time:

```text
run 1: 0.729 s
run 2: 0.723 s
run 3: 0.723 s
mean : 0.725 s
```

The serial time measures the escape-time loop only; it excludes the final PPM
write and hash. The parallel time includes `MPI_Gatherv`, so the reported
parallel speed-up is slightly conservative.

## 4. MPI row decomposition

### 4.1 Partition formula

For `P` MPI ranks and `H` image rows:

```text
base = H / P
rem  = H % P
local_h = base + (rank < rem ? 1 : 0)
y0      = rank * base + min(rank, rem)
```

The first `rem` ranks receive one extra row. For `H = 600` and `P = 4`:

| Rank | y0 | Rows | End |
|---|---:|---:|---:|
| 0 | 0 | 150 | 150 |
| 1 | 150 | 150 | 300 |
| 2 | 300 | 150 | 450 |
| 3 | 450 | 150 | 600 |

Each rank allocates a compact buffer of:

```text
local_h * width * 3 bytes
```

and renders only its own rows.

### 4.2 No halo exchange

Every output pixel belongs to exactly one rank. There is no neighbour data, no
ghost row and no iteration between ranks. The only MPI communication is:

1. three small broadcasts of `width`, `height` and `max_iter`;
2. one `MPI_Gatherv` at the end.

This is why the communication cost is small compared with the rendering work.

### 4.3 Broadcasting parameters

Rank 0 parses the command line and broadcasts the three integers:

```c
MPI_Bcast(&width,    1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&height,   1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
```

All ranks must agree on the image size and iteration limit; otherwise the
buffers and the row split would not match.

### 4.4 Gathering the image

After rendering, rank 0 receives all bands with `MPI_Gatherv`.

For each rank `r`, the code computes:

```text
rows_r   = base + (r < rem ? 1 : 0)
counts[r] = rows_r * width * 3
displs[r] = sum of counts[0..r-1]
```

`counts` is in **bytes**, not rows, because `MPI_Gatherv` counts elements of
the send buffer, and each element is one RGB byte. `displs` is also in bytes
and points to the start of that band in the final image.

The root allocates the complete image:

```c
full = malloc(width * height * 3);
```

and calls:

```c
MPI_Gatherv(local, local_bytes, MPI_UNSIGNED_CHAR,
            full, counts, displs, MPI_UNSIGNED_CHAR,
            0, MPI_COMM_WORLD);
```

Rank 0 then computes the FNV hash, writes the PPM and prints the summary.

### 4.5 Timing

All ranks call `MPI_Barrier` before `t0 = MPI_Wtime()`. The timed region
contains:

- local rendering;
- `MPI_Gatherv`.

It excludes the final PPM write. The rank-0 `t1 - t0` is used as the wall time.

## 5. OpenMP layer

Inside a rank, `render_rows` parallelises the row loop:

```c
#pragma omp parallel for
for (int j = 0; j < nrows; ++j) {
    ...
}
```

Each thread gets a private `j`, computes its own `ci`, and writes a disjoint
part of the local buffer. There is no shared accumulator and no reduction, so
the loop is race-free.

### 5.1 Static schedule

```c
#pragma omp parallel for schedule(static)
```

The iterations are divided into contiguous blocks at the start. With 600 rows
and 4 threads, each thread receives 150 rows.

### 5.2 Dynamic schedule

```c
#pragma omp parallel for schedule(dynamic, 8)
```

A thread claims eight rows at a time. When it finishes, it claims the next
eight. This allows a thread that received cheap rows to take over part of the
expensive middle region.

Eight rows is a compromise:

- smaller chunks balance more finely but add scheduling overhead;
- larger chunks reduce overhead but can leave a long tail.

## 6. Determinism and floating point

The renderer has no random number generator. A pixel's colour is a pure
function of its coordinates and `max_iter`, so the order in which threads or
ranks process pixels cannot change the result.

The build uses:

```text
-ffp-contract=off
```

This prevents the compiler from fusing a multiply and an add into an FMA
instruction. Without it, two builds can differ by one iteration on a boundary
pixel, which changes the colour and the hash even though the algorithm is the
same.

## 7. Build

Requirements:

- Ubuntu Server 24.04 (or another Linux with GCC and OpenMPI);
- GCC with OpenMP;
- OpenMPI (`mpicc`, `mpirun`);
- optional: ImageMagick and `chafa` for viewing the PPM in a terminal.

Install:

```bash
sudo apt update
sudo apt install build-essential libopenmpi-dev openmpi-bin
```

Build:

```bash
make
```

The Makefile produces:

```text
mandelbrot_serial
mandelbrot_static
mandelbrot_dynamic
```

Important flags:

```text
-O2 -std=c11 -Wall -Wextra -fopenmp -ffp-contract=off -lm
```

## 8. Run

### 8.1 Full experiment

```bash
./scripts/run.sh
```

The script runs:

- serial 1x1;
- static 1x4 and 2x2;
- dynamic 1x4, 2x2 and 4x1;

then prints the SHA-256 of all output images.

### 8.2 One configuration manually

```bash
OMP_NUM_THREADS=2 mpirun --bind-to none -x OMP_NUM_THREADS \
  -np 2 ./mandelbrot_dynamic 800 600 1000 out/2x2.ppm
```

`--bind-to none` prevents OpenMPI from pinning all of a rank's threads to one
core. `-x OMP_NUM_THREADS` forwards the thread count to every rank.

### 8.3 Command line

```text
./mandelbrot_dynamic [width] [height] [max_iter] [output.ppm]
```

| Argument | Default | Meaning |
|---|---:|---|
| width | 800 | image width in pixels |
| height | 600 | image height in pixels |
| max_iter | 1000 | iteration limit |
| output.ppm | mandelbrot.ppm | output file |

`mandelbrot_serial` and `mandelbrot_static` accept the same arguments.

### 8.4 Output

Each rank prints its row range, then rank 0 prints the summary:

```text
rank 0/4: y0=0 rows=150
rank 1/4: y0=150 rows=150
...
MPI+OpenMP Mandelbrot 800x600 max_iter=1000 ranks=4 threads/rank=1
time        : 0.310 s
FNV-1a hash : 92193b63bbd42a23
output      : out/4x1_dynamic.ppm
```

`time` contains rendering plus `MPI_Gatherv`; it excludes the file write. The
hash is computed by rank 0 from the assembled image.

## 9. Results

Measured on a VMware Fusion Ubuntu Server 24.04 VM with **4 vCPUs**,
800 x 600, `max_iter = 1000`, plus a two-node check on the lab cluster.

### 9.1 Static baseline

These are single observations, used as the static comparison.

| Configuration | Workers | Time (s) | Speed-up | Efficiency |
|---|---:|---:|---:|---:|
| serial 1x1 | 1 | 0.725 | 1.00 | 1.000 |
| static 1x4 | 4 | 0.309 | 2.35 | 0.587 |
| static 2x2 | 4 | 0.326 | 2.22 | 0.556 |
| static 4x1 | 4 | 0.310 | 2.34 | 0.585 |

### 9.2 Dynamic schedule

These are the means of three runs.

| Configuration | Run 1 | Run 2 | Run 3 | Mean | Speed-up | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| dynamic 1x4 | 0.183 | 0.183 | 0.181 | 0.1823 | 3.98 | 0.994 |
| dynamic 2x2 | 0.184 | 0.182 | 0.184 | 0.1833 | 3.95 | 0.989 |
| dynamic 4x1 | 0.309 | 0.310 | 0.309 | 0.3093 | 2.34 | 0.586 |

Speed-up and efficiency are defined as:

```text
S(W) = T(1) / T(W)
E(W) = S(W) / W
```

where `W` is the total number of workers (ranks multiplied by threads per
rank). The serial mean is the `T(1)` baseline.

### 9.3 Two-node check

The same binaries were also run on two compute nodes. The first configuration
used one rank per node with two OpenMP threads; the second used two ranks per
node with one thread each.

| Configuration | Layout | Workers | Mean time (s) | Speed-up | Efficiency |
|---|---|---:|---:|---:|---:|
| serial on headnode | 1 rank, 1 thread | 1 | 0.416 | 1.00 | 1.000 |
| 2x2 across two nodes | 1 rank/node, 2 threads | 4 | 0.1193 | 3.49 | 0.872 |
| 4x1 across two nodes | 2 ranks/node, 1 thread | 4 | 0.1977 | 2.10 | 0.526 |

The 2x2 hybrid is 1.66x faster than 4x1 even though both use four workers. This
repeats the single-node pattern: the hybrid gives each rank two threads that can
share the expensive rows, while 4x1 has one thread per rank and still pays for
the static MPI band split. All runs produced the same FNV hash.

The multi-node times are not compared with the single-VM 0.725 s baseline: the
compute nodes and the VM have different CPU allocation and load. Within the
cluster environment, the 0.416 s serial mean is the correct baseline.

## 10. Why the results are what they are

### 10.1 The workload is not uniform

Rows through the centre of the image cross the Mandelbrot set and run to
`max_iter`. Rows near the top and bottom escape almost immediately.

A row-cost measurement of the same 800 x 600 image gives:

- central row (row 300): about **612 iterations per pixel**;
- outer rows: about **3 iterations per pixel**.

That is a difference of more than two orders of magnitude.

The image is therefore not a uniform workload. A scheduler that gives each
worker a fixed contiguous block can give one worker most of the expensive
rows.

### 10.2 Why static scheduling stops at about 2.3x

All three static configurations have a 150-row granularity:

| Configuration | How 600 rows are split | Smallest static block |
|---|---|---:|
| static 1x4 | 600 / 4 threads | 150 rows |
| static 2x2 | 600 / 2 ranks, then 300 / 2 threads | 150 rows |
| static 4x1 | 600 / 4 ranks | 150 rows |

The slowest 150-row block contains the expensive middle rows and takes about
0.31 s. Other workers finish earlier and wait. The wall time is set by that
block:

```text
0.725 / 0.310 ≈ 2.34x
```

This explains why all static configurations land around 2.2-2.35x, even though
they use four workers.

The static 2x2 result (0.326 s) is slightly worse than 1x4 (0.309 s) because
the two 300-row bands still create a rank-level split, and the final gather adds
a small cost. The difference is small.

### 10.3 Why dynamic scheduling reaches about 3.98x

With `schedule(dynamic, 8)`, a free thread immediately claims another eight
rows. The expensive rows are no longer owned by one thread for the whole run;
they are shared among the threads that become free.

The ideal time with four workers is:

```text
0.725 / 4 = 0.181 s
```

The measured 1x4 mean is 0.1823 s, only about 0.6% above the ideal. The dynamic
schedule therefore removes almost all of the load imbalance.

### 10.4 Why 1x4 and 2x2 are almost tied

Both use four threads and both spread the expensive rows. In 2x2, each rank
owns a fixed 300-row band, but the top and bottom bands each contain roughly
half of the expensive central region, so the two ranks have similar work.

The dynamic schedule balances the two threads inside each rank, and the
remaining rank-level difference is small. The 0.5% gap between 0.1823 s and
0.1833 s is within run-to-run variation, so the two configurations are
effectively tied on this VM.

### 10.5 Why 4x1 stays at 2.34x

4x1 uses the same dynamic binary, but:

```bash
OMP_NUM_THREADS=1
```

Each MPI rank has only one thread. `schedule(dynamic)` has no second thread to
give work to, so the schedule has no effect.

The 4x1 time remains 0.3093 s, almost identical to the static 4x1 value of
0.310 s. This is the control experiment:

- if dynamic scheduling were the cause of the speed-up, 4x1 should not improve;
- it does not improve;
- therefore the improvement in 1x4 and 2x2 comes from OpenMP dynamic
  scheduling, not from recompiling or from MPI itself.

### 10.6 Why the speed-up is not exactly 4.00x

Even the best configuration has a small gap:

- `MPI_Gatherv` and the final barrier;
- OpenMP thread start-up and scheduling;
- a residual tail at the end of the image;
- VM vCPUs sharing physical host cores;
- only three repetitions, so measurement noise remains;
- the serial timing excludes the PPM write while the parallel timing includes
  the gather, which slightly penalises the parallel result.

The measured 99% efficiency should therefore be read as "close to ideal in
this particular four-vCPU measurement", not as a universal claim.

## 11. Correctness check

```bash
./scripts/run.sh
```

At the end, all six SHA-256 lines must be identical. The expected digest is:

```text
4606e1e408496266fd928852c8b0c4137113773c7d01f4b814b3ad058827c17e
```

If a hash differs, a row has been lost, duplicated or written to the wrong
offset. The deterministic pixel function and `-ffp-contract=off` make this a
strict check.

## 12. Repository layout

```text
.
├── Makefile
├── README.md
├── README.zh-CN.md
├── docs/
│   └── mandelbrot.png
├── scripts/
│   └── run.sh
├── src/
│   ├── mandelbrot_serial.c
│   ├── mandelbrot_mpi_omp.c
│   └── mandelbrot_mpi_omp_static.c
├── steps/
│   ├── README.md
│   ├── Makefile
│   ├── hello_mpi.c
│   ├── mpi_rows.c
│   ├── mandelbrot_serial.c
│   ├── mandelbrot_mpi.c
│   ├── mandelbrot_mpi_omp_static.c
│   └── mandelbrot_mpi_omp_dynamic.c
└── lab9_amd/
    ├── Lab9_Mandelbrot_AMD_ROCm.ipynb
    ├── Lab9_Mandelbrot_AMD_ROCm_ZH.ipynb
    └── README.md
```

- `mandelbrot_serial.c`: serial reference.
- `mandelbrot_mpi_omp.c`: final hybrid version with dynamic scheduling.
- `mandelbrot_mpi_omp_static.c`: same hybrid code with static scheduling.
- `scripts/run.sh`: runs the full measurement matrix and compares hashes.
- `steps/`: the incremental programs from the beginning of the lab: MPI hello,
  row partition, serial renderer, pure MPI, static hybrid and dynamic hybrid.
  See [`steps/README.md`](steps/README.md) for the step-by-step commands.
- `README.zh-CN.md`: Chinese translation of this document.
- `lab9_amd/`: Jupyter notebook for running the C project and a PyTorch ROCm
  GPU port on the AMD cluster.

## 13. Reproduce from scratch

```bash
git clone https://gitcode.com/JiakunPi/mandelbrot-mpi-openmp.git
cd mandelbrot-mpi-openmp
make
./scripts/run.sh
```

The final hashes should all match the digest above.

## 14. Limitations and future work

- The scaling study is single-node; the two-node check confirms placement and
  correctness, but it is not a full multi-node scaling study.
- MPI uses contiguous row bands. A cyclic row assignment or a dynamic MPI tile
  queue could balance the ranks too, at the cost of more complex gathering and
  more messages.
- The image is small (1.4 MB), so the measured efficiency is optimistic for
  larger images.
- Static measurements are single observations; dynamic and serial values are
  three-run means.
- Possible extensions: larger images, AVX/SIMD inner loops, cost-aware tile
  scheduling, and a hybrid MPI tile queue.

## 15. Report

The full laboratory report is submitted separately and is not stored in this
public repository because it contains personal information.

## 16. AMD ROCm notebook

The `lab9_amd/` directory contains a self-contained notebook for the AMD ROCm
cluster. It checks the environment, runs the C serial/static/dynamic programs
when `mpicc` and `mpirun` are available, then ports the same Mandelbrot
algorithm to PyTorch ROCm on the AMD GPU.

- [English notebook](lab9_amd/Lab9_Mandelbrot_AMD_ROCm.ipynb)
- [Chinese notebook](lab9_amd/Lab9_Mandelbrot_AMD_ROCm_ZH.ipynb)
- [Instructions](lab9_amd/README.md)

The notebook compares the AMD container with the original four-vCPU VM numbers
(serial 0.725 s, dynamic 2x2 0.1833 s). The cross-environment ratio is only a
reference; the controlled comparison is C serial vs C MPI+OpenMP vs GPU inside
the AMD container.
