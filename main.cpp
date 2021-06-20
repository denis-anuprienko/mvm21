// gcc -O2 mvm.c -lm
// icc -O2 mvm.c -lm
// ./a.out

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const int M = 100;            // just a number of repetitions

int main (int argc, char *argv[])
{
    if(argc != 2){
        printf("usage: main <matrix_size>\n");
        exit(1);
    }
    int N = atoi(argv[1]);
    int i, j, k, n, id, np, th;
    double *a, *x, *y, *X, *ai, T, perf, sum;

    id = 0; np = 1; th = 1; // just a test for 1 process and 1 thread
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &id);
    MPI_Comm_size (MPI_COMM_WORLD, &np);
    #ifdef USE_OMP
    th = omp_get_max_threads();
    #endif

    n = N / np; // local number of vector components and matrix rows
    a = new double[static_cast<size_t>(n*N)]; // local rows of matrix A
    x = new double[static_cast<size_t>(n)];   // local components of vector x
    y = new double[static_cast<size_t>(n)];   // local components of vector y
    X = new double[static_cast<size_t>(N)];   // working vector with global X

    // Initialize owned part of the matrix
    for(i = 0; i < n; i++)
        for(j = 0; j < N; j++)
            a[(i+id*n)*N + j] = 1. / (1. + i+id*n + j); // global i and j

    // Initialize owned parts of vectors
    for(i = 0; i < n; i++){
        x[i] = i+id*n;
        y[i] = 0.;
    }

    //MPI_Barrier(MPI_COMM_WORLD);
    T = MPI_Wtime();
    for(k = 0; k < M; k++){

        //...some code to collect global X[] from the other MPI processes

        // omp parallel for pragma should be placed here!
#pragma omp parallel for
        for(i = 0; i < n; i++){
            ai = a + i*N; // address of i-th matrix row
            for (j = 0; j < N; j++)
                y[i] += ai[j] * x[j]; // use global X[] instead of x[] here!
        }
    }
    //MPI_Barrier(MPI_COMM_WORLD);
    T = MPI_Wtime() - T;
    perf = 1e-6 * N * N * M / T;

    // Compute norm of vector Y here: sum=||Y||
    sum = 0.0;
    // omp parallel for pragma with "reduction(+:sum)" should be placed here
#ifdef USE_OMP
#pragma omp parallel for reduction(+:sum)
#endif
    for(i = 0; i < n; i++)
        sum += y[i] * y[i];
    //MPI_Allreduce(...) for sum should be placed here
    sum = sqrt(sum) / M;

    if (id == 0)
       printf("MPI-1p mvm: N=%d np=%d th=%d norm=%lf time=%lf perf=%.2lf MFLOPS\n", N, np, th, sum, T, perf);

    MPI_Finalize();
    return 0;
}
