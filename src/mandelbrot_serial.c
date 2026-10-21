#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>

#define DEFAULT_WIDTH    800
#define DEFAULT_HEIGHT   600
#define DEFAULT_MAX_ITER 1000

static double now_seconds(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

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

int main(int argc, char **argv)
{
    int width    = (argc > 1) ? atoi(argv[1]) : DEFAULT_WIDTH;
    int height   = (argc > 2) ? atoi(argv[2]) : DEFAULT_HEIGHT;
    int max_iter = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITER;
    const char *out = (argc > 4) ? argv[4] : "mandelbrot.ppm";

    if (width < 16 || height < 16 || max_iter < 1) {
        fprintf(stderr, "usage: %s [width] [height] [max_iter] [out.ppm]\n", argv[0]);
        return 1;
    }

    unsigned char *pixels = malloc((size_t)width * (size_t)height * 3u);
    if (!pixels) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    const double xmin = -2.10;
    const double xmax =  0.70;
    const double xspan = xmax - xmin;
    const double yspan = xspan * (double)height / (double)width;
    const double ymin = -0.5 * yspan;
    const double ymax =  0.5 * yspan;

    double t0 = now_seconds();

    for (int py = 0; py < height; ++py) {
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

            size_t idx = ((size_t)py * (size_t)width + (size_t)px) * 3u;
            pixels[idx + 0] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
        }
    }

    double elapsed = now_seconds() - t0;
    uint64_t hash = fnv1a(pixels, (size_t)width * (size_t)height * 3u);

    if (!write_ppm(out, pixels, width, height)) {
        free(pixels);
        return 1;
    }

    printf("Mandelbrot %dx%d max_iter=%d\n", width, height, max_iter);
    printf("serial time : %.3f s\n", elapsed);
    printf("FNV-1a hash : %016" PRIx64 "\n", hash);
    printf("output      : %s\n", out);

    free(pixels);
    return 0;
}
