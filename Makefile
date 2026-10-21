CC       ?= gcc
MPICC    ?= mpicc
CFLAGS   ?= -O2 -std=c11 -Wall -Wextra -ffp-contract=off
OMPFLAGS ?= -fopenmp
LDLIBS   ?= -lm

all: mandelbrot_serial mandelbrot_dynamic mandelbrot_static

mandelbrot_serial: src/mandelbrot_serial.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

mandelbrot_dynamic: src/mandelbrot_mpi_omp.c
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS)

mandelbrot_static: src/mandelbrot_mpi_omp_static.c
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS)

run: all
	./scripts/run.sh

clean:
	rm -f mandelbrot_serial mandelbrot_dynamic mandelbrot_static
	rm -rf out

.PHONY: all run clean
