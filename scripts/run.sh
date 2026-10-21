#!/usr/bin/env bash
set -euo pipefail

MPIRUN="${MPIRUN:-mpirun}"
MPIRUN_ARGS="${MPIRUN_ARGS:---bind-to none}"
W="${W:-800}"
H="${H:-600}"
ITER="${ITER:-1000}"

mkdir -p out

run_mpi() {
    local ranks="$1" threads="$2" binary="$3" tag="$4"
    echo
    echo ">>> ${ranks} MPI rank(s) x ${threads} OpenMP thread(s): ${tag}"
    # shellcheck disable=SC2086
    OMP_NUM_THREADS="$threads" "$MPIRUN" $MPIRUN_ARGS \
        -x OMP_NUM_THREADS -np "$ranks" "./$binary" \
        "$W" "$H" "$ITER" "out/${tag}.ppm"
}

echo ">>> serial reference"
./mandelbrot_serial "$W" "$H" "$ITER" out/serial.ppm

run_mpi 1 4 mandelbrot_static  1x4_static
run_mpi 2 2 mandelbrot_static  2x2_static
run_mpi 1 4 mandelbrot_dynamic 1x4_dynamic
run_mpi 2 2 mandelbrot_dynamic 2x2_dynamic
run_mpi 4 1 mandelbrot_dynamic 4x1_dynamic

echo
echo ">>> image hashes"
sha256sum out/serial.ppm \
          out/1x4_static.ppm out/2x2_static.ppm \
          out/1x4_dynamic.ppm out/2x2_dynamic.ppm out/4x1_dynamic.ppm
