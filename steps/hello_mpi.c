#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    int rank, size, len;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Get_processor_name(hostname, &len);

    /* The print order is not guaranteed between ranks. */
    printf("Hello from rank %d of %d on %s\n", rank, size, hostname);

    MPI_Finalize();
    return 0;
}
