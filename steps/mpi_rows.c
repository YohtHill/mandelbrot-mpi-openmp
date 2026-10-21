#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Rank 0 reads the command line and broadcasts it to everyone else. */
    int height = 600;
    if (rank == 0 && argc > 1)
        height = atoi(argv[1]);

    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (height < 1) {
        if (rank == 0)
            fprintf(stderr, "height must be positive\n");
        MPI_Finalize();
        return 1;
    }

    /* Split the rows as evenly as possible between the ranks. */
    int base = height / size;
    int rem  = height % size;

    /* The first rem ranks receive one extra row. */
    int local_h = base + (rank < rem ? 1 : 0);

    /* Skip the extra rows that belong to the ranks before this one. */
    int y0 = rank * base + (rank < rem ? rank : rem);

    printf("rank %d/%d: y0=%d rows=%d end=%d\n",
           rank, size, y0, local_h, y0 + local_h);

    MPI_Finalize();
    return 0;
}
