#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#define DEFAULT_WIDTH    800
#define DEFAULT_HEIGHT   600
#define DEFAULT_MAX_ITER 1000

static uint64_t fnv1a(const unsigned char *data, size_t len)
{
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < len; ++i) {
        h ^= (uint64_t)data[i];
        h *= 1099511628211ull;
    }
    return h;
}

static int write_ppm(const char *path, const unsigned char *pixels,
                     int width, int height)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return 0;
    }
    fprintf(f, "P6\n%d %d\n255\n", width, height);

    size_t bytes = (size_t)width * (size_t)height * 3u;
    if (fwrite(pixels, 1, bytes, f) != bytes) {
        fprintf(stderr, "short write to %s\n", path);
        fclose(f);
        return 0;
    }
    fclose(f);
    return 1;
}

/* Render nrows rows starting at y0 into a compact RGB buffer. */
static void render_rows(int y0, int nrows, int width, int height,
                        int max_iter, unsigned char *out)
{
    const double xmin = -2.10;
    const double xmax =  0.70;
    const double xspan = xmax - xmin;
    const double yspan = xspan * (double)height / (double)width;
    const double ymin = -0.5 * yspan;
    const double ymax =  0.5 * yspan;

    for (int j = 0; j < nrows; ++j) {
        int py = y0 + j;
        double ci = ymax - (ymax - ymin) * ((double)py + 0.5) / (double)height;

        for (int px = 0; px < width; ++px) {
            double cr = xmin + (xmax - xmin) * ((double)px + 0.5) / (double)width;

            double zr = 0.0;
            double zi = 0.0;
            int iter = 0;

            while (iter < max_iter && zr * zr + zi * zi <= 4.0) {
                double next_r = zr * zr - zi * zi + cr;
                double next_i = 2.0 * zr * zi + ci;
                zr = next_r;
                zi = next_i;
                ++iter;
            }

            unsigned char r, g, b;
            if (iter == max_iter) {
                r = g = b = 0;
            } else {
                double t = sqrt((double)iter / 200.0);
                if (t > 1.0)
                    t = 1.0;
                double rr = fmin(1.0, 2.0 * t);
                double gg = fmin(1.0, 2.0 * t - 0.5);
                double bb = fmin(1.0, 1.0 - 2.0 * t);
                r = (unsigned char)(255.0 * rr);
                g = (unsigned char)(255.0 * gg);
                b = (unsigned char)(255.0 * bb);
            }

            size_t idx = ((size_t)j * (size_t)width + (size_t)px) * 3u;
            out[idx + 0] = r;
            out[idx + 1] = g;
            out[idx + 2] = b;
        }
    }
}

int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width    = DEFAULT_WIDTH;
    int height   = DEFAULT_HEIGHT;
    int max_iter = DEFAULT_MAX_ITER;
    const char *out = "mpi_mandelbrot.ppm";

    if (rank == 0) {
        if (argc > 1) width    = atoi(argv[1]);
        if (argc > 2) height   = atoi(argv[2]);
        if (argc > 3) max_iter = atoi(argv[3]);
        if (argc > 4) out      = argv[4];
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (width < 16 || height < 16 || max_iter < 1) {
        if (rank == 0)
            fprintf(stderr, "usage: %s [width] [height] [max_iter] [out.ppm]\n",
                    argv[0]);
        MPI_Finalize();
        return 1;
    }

    /* Divide the rows between the ranks. */
    int base = height / size;
    int rem  = height % size;
    int local_h = base + (rank < rem ? 1 : 0);
    int y0 = rank * base + (rank < rem ? rank : rem);

    size_t local_bytes = (size_t)local_h * (size_t)width * 3u;
    unsigned char *local = malloc(local_bytes);
    if (!local) {
        fprintf(stderr, "rank %d: out of memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Counts and displacements are needed by MPI_Gatherv on rank 0. */
    int *counts = malloc((size_t)size * sizeof(int));
    int *displs = malloc((size_t)size * sizeof(int));
    if (!counts || !displs) {
        fprintf(stderr, "rank %d: out of memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int offset = 0;
    for (int r = 0; r < size; ++r) {
        int rows = base + (r < rem ? 1 : 0);
        counts[r] = rows * width * 3;
        displs[r] = offset;
        offset += counts[r];
    }

    unsigned char *full = NULL;
    if (rank == 0) {
        full = malloc((size_t)width * (size_t)height * 3u);
        if (!full) {
            fprintf(stderr, "rank 0: out of memory\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    printf("rank %d/%d: y0=%d rows=%d\n", rank, size, y0, local_h);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    render_rows(y0, local_h, width, height, max_iter, local);

    MPI_Gatherv(local, (int)local_bytes, MPI_UNSIGNED_CHAR,
                full, counts, displs, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    if (rank == 0) {
        uint64_t hash = fnv1a(full, (size_t)width * (size_t)height * 3u);
        if (!write_ppm(out, full, width, height)) {
            free(local);
            free(counts);
            free(displs);
            free(full);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("MPI Mandelbrot %dx%d max_iter=%d ranks=%d\n",
               width, height, max_iter, size);
        printf("time        : %.3f s\n", t1 - t0);
        printf("FNV-1a hash : %016" PRIx64 "\n", hash);
        printf("output      : %s\n", out);
    }

    free(local);
    free(counts);
    free(displs);
    free(full);

    MPI_Finalize();
    return 0;
}
