# Lab 9 on the AMD ROCm cluster

This folder contains a self-contained Jupyter notebook that runs the Lab 9
Mandelbrot project on the AMD ROCm cluster.

## Files

- `Lab9_Mandelbrot_AMD_ROCm.ipynb` — fully English notebook (recommended for submission).
- `Lab9_Mandelbrot_AMD_ROCm_ZH.ipynb` — Chinese version of the same notebook.

## What the notebook does

1. Checks the Python, compiler, MPI and GPU environment.
2. Writes the final C sources (`serial`, `static`, `dynamic`) into
   `lab9_amd_work/`.
3. Compiles and runs the C serial program, and the MPI + OpenMP static/dynamic
   versions if `mpicc` and `mpirun` exist.
4. Ports the same Mandelbrot algorithm to PyTorch ROCm.
5. Compares the C and GPU results, saves PPM/PNG images and writes
   `results.json`.
6. Compares the AMD container with the original four-vCPU VM numbers
   (serial 0.725 s, dynamic 2x2 0.1833 s). The cross-environment ratio is
   reported only as a reference; the controlled comparison is inside the
   AMD container.

## How to run

1. Upload `Lab9_Mandelbrot_AMD_ROCm.ipynb` to the JupyterLab environment
   (for example, next to `GPU_Usage.ipynb`).
2. Open it and select the Python 3 kernel that has PyTorch ROCm.
3. Run all cells.

The notebook creates a working directory:

```text
lab9_amd_work/
├── mandelbrot_serial.c
├── mandelbrot_mpi_omp.c
├── mandelbrot_mpi_omp_static.c
├── c_serial.ppm
├── c_static_2x2.ppm
├── c_dynamic_2x2.ppm
├── gpu_mandelbrot.ppm
├── gpu_mandelbrot.png
└── results.json
```

## Notes

- `torch.cuda` is the PyTorch API used for both CUDA and ROCm. On this cluster
  it maps to the AMD GPU; check `torch.version.hip` in the first cell.
- If `mpicc` or `mpirun` is not installed in the Jupyter container, the MPI
  section is skipped and the GPU section still runs.
- The Jupyter container is a single node. The two-node cluster evidence is in
  the report; this notebook is a single-container MPI run plus a GPU port.
- The C code uses `-ffp-contract=off`, so the C hash should be reproducible
  with the same compiler and libm. A different compiler, libm or CPU/GPU
  architecture can change boundary pixels; compare hashes within the same
  environment.
- The GPU version uses `float64` by default. If the device cannot run it, the
  notebook falls back to `float32`; in that case the pixel differences from the
  C `double` version will be larger.

## Full project

- GitHub: <https://github.com/YohtHill/mandelbrot-mpi-openmp>
- GitCode: <https://gitcode.com/JiakunPi/mandelbrot-mpi-openmp>
