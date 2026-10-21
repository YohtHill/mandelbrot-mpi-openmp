# Step-by-step build for the Lab 9 development history.
#
# These are the intermediate programs used while developing the final
# renderer.  The final project is in ../src and ../Makefile.

CC       ?= gcc
MPICC    ?= mpicc
CFLAGS   ?= -O2 -std=c11 -Wall -Wextra -ffp-contract=off
OMPFLAGS ?= -fopenmp
LDLIBS   ?= -lm

all: hello_mpi mpi_rows mandelbrot_serial mandelbrot_mpi \
     mandelbrot_mpi_omp_static mandelbrot_mpi_omp_dynamic

hello_mpi: hello_mpi.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDLIBS)

mpi_rows: mpi_rows.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDLIBS)

mandelbrot_serial: mandelbrot_serial.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

mandelbrot_mpi: mandelbrot_mpi.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDLIBS)

mandelbrot_mpi_omp_static: mandelbrot_mpi_omp_static.c
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS)

mandelbrot_mpi_omp_dynamic: mandelbrot_mpi_omp_dynamic.c
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f hello_mpi mpi_rows mandelbrot_serial mandelbrot_mpi \
	      mandelbrot_mpi_omp_static mandelbrot_mpi_omp_dynamic

.PHONY: all clean
