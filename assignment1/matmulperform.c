/******************************************************************************
* FILE: matmulperform.c (formerly mpi_mm.c)
* DESCRIPTION:  
*   MPI Matrix Multiply - C Version
*   In this code, the master task distributes a matrix multiply
*   operation to numtasks-1 worker tasks.
*   NOTE:  C and Fortran versions of this code differ because of the way
*   arrays are stored/passed.  C arrays are row-major order but Fortran
*   arrays are column-major order.
* AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
*   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
* LAST REVISED: 04/13/05
* Modified to randomly generate square matrices
* and distribute both rows and cols: Mengxiang Jiang 10/18/22
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define DEBUG false            /* debug flag for printing debug statments */
#define MAXN 500               /* max row/col size */
#define MAXRUN 10              /* number of runs */
#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int main (int argc, char *argv[])
{
    int     numtasks,              /* number of tasks in partition */
            taskid,                /* a task identifier */
            numworkers,            /* number of worker tasks */
            source,                /* task id of message source */
            dest,                  /* task id of message destination */
            mtype,                 /* message type */
            rows,                  /* rows of matrix A sent to each worker */
            averow, extra, offset, /* used to determine rows sent to each worker */
            i, j, k, rc,           /* misc */
            n,                     /* number of rows and columns of all matrices */
            runs;                  /* for keeping track of runs for a given n */
    double  startwtime, endwtime,  /* used to measure how long the calculation takes*/
            totaltime, avgtime;    /* used to aggregate times for different matrix sizes*/            
    bool    a_done, b_done;        /* used to indicate whether processing of respective matrix is done */
    MPI_Status status;             /* used to store status from MPI */

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    if (numtasks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    numworkers = numtasks-1;

    srand(time(NULL));
    FILE *output = fopen("output.csv", "w");

    for (n=2; n <= MAXN; n++)
    {
        totaltime = 0;
        for (runs=0; runs<MAXRUN; runs++)
        {
/**************************** master task ************************************/
            if (taskid == MASTER)
            {
                startwtime = MPI_Wtime();
                #if DEBUG
                printf("mpi_mm has started with %d tasks.\n",numtasks);
                printf("Initializing matrices...\n");
                #endif

                double  a[n][n],    /* matrix A to be multiplied */
                        b[n][n],    /* matrix B to be multiplied (rows and columns swapped for message passing) */
                        c[n][n];    /* result matrix C */
                
                /* matrix entries are doubles in the range from -1 to 1*/
                for (i=0; i<n; i++)
                {
                    for(j=0; j<n; j++)
                    {
                        a[i][j] = 2 * (0.5 - (double)(rand())/RAND_MAX);
                        b[i][j] = 2 * (0.5 - (double)(rand())/RAND_MAX);
                    }
                }

                #if DEBUG
                printf("\nMatrix A:\n");
                for (i=0; i<n; i++)
                {
                    printf("\n"); 
                    for (j=0; j<n; j++) 
                        printf("%6.2f   ", a[i][j]);
                }
                printf("\n******************************************************\n");

                printf("\nMatrix B:\n");
                for (i=0; i<n; i++)
                {
                    printf("\n"); 
                    for (j=0; j<n; j++) 
                        printf("%6.2f   ", b[j][i]);
                }
                printf("\n******************************************************\n");
                #endif

                /* Send matrix data to the worker tasks */
                averow = n/numworkers;
                extra = n%numworkers;
                offset = 0;
                mtype = FROM_MASTER;
                for (dest=1; dest<=numworkers; dest++)
                {
                    rows = (dest <= extra) ? averow+1 : averow;
                    #if DEBUG   	
                    printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
                    #endif
                    MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
                    MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
                    MPI_Send(&n, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
                    MPI_Send(&n, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
                    MPI_Send(&a[offset][0], rows*n, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
                    MPI_Send(&b, n*n, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
                    offset = offset + rows;
                }

                /* Receive results from worker tasks */
                mtype = FROM_WORKER;
                for (i=1; i<=numworkers; i++)
                {
                    source = i;
                    MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
                    MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
                    MPI_Recv(&c[offset][0], rows*n, MPI_DOUBLE, source, mtype, 
                            MPI_COMM_WORLD, &status);
                    #if DEBUG  
                    printf("Received results from task %d\n",source);
                    #endif
                }
                endwtime = MPI_Wtime();
                
                #if DEBUG
                /* Print results */
                printf("******************************************************\n");
                printf("Result Matrix:\n");
                for (i=0; i<n; i++)
                {
                    printf("\n"); 
                    for (j=0; j<n; j++) 
                        printf("%6.2f   ", c[i][j]);
                }
                printf("\n******************************************************\n");
                printf ("Master done.\n");
                printf("wall clock time = %f\n", endwtime-startwtime);
                #endif
                totaltime += (endwtime - startwtime);
            }

        /**************************** worker task ************************************/
            if (taskid > MASTER)
            {
                mtype = FROM_MASTER;
                MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
                MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
                MPI_Recv(&n, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
                MPI_Recv(&n, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
                double a[rows][n];
                double b[n][n];
                double c[rows][n];

                MPI_Recv(&a, rows*n, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
                MPI_Recv(&b, n*n, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

                for (k=0; k<n; k++)
                {
                    for (i=0; i<rows; i++)
                    {
                        c[i][k] = 0.0;
                        for (j=0; j<n; j++)
                            c[i][k] = c[i][k] + a[i][j] * b[k][j];
                    }
                }
                mtype = FROM_WORKER;
                MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
                MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
                MPI_Send(&c, rows*n, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
            }
        }
        if (taskid == MASTER)
        {
            avgtime = totaltime/MAXRUN;
            printf("%i,%f\n", n, avgtime);
            fprintf(output, "%i,%f\n", n, avgtime);
        }
    }
    fclose(output);
    MPI_Finalize();
}