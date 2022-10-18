/******************************************************************************
* FILE: matrixmul.c (formerly mpi_mm.c)
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
* Modified to read from file and pass single columns of matrix B: 
* Mengxiang Jiang 10/22
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define DEBUG true             /* debug flag for printing debug statments */
#define MAXCHAR 1024           /* max number of characters per line in input file */
#define MAXROW 100             /* max number of rows in a matrix */
#define MAXCOL 100             /* max number of cols in a matrix */
#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int main (int argc, char *argv[])
{
    if (argc != 2)                 /* needs a input file for matrix initialization */
    {
        printf("usage: mpirun ./matrixmul -np 4 filename");
        exit(1);
    }

    int     numtasks,              /* number of tasks in partition */
            taskid,                /* a task identifier */
            numworkers,            /* number of worker tasks */
            source,                /* task id of message source */
            dest,                  /* task id of message destination */
            mtype,                 /* message type */
            rows,                  /* rows of matrix A sent to each worker */
            averow, extra, offset, /* used to determine rows sent to each worker */
            i, j, k, rc,           /* misc */
            nra,                   /* number of rows in the first matrix */
            nca,                   /* number of columns in the first matrix (same as rows in the second) */
            ncb;                   /* number of columns in the second matrix */
    double  startwtime, endwtime;  /* used to measure how long the calculation takes*/
    MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    if (numtasks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    numworkers = numtasks-1;


/**************************** master task ************************************/
    if (taskid == MASTER)
    {
        startwtime = MPI_Wtime();
        printf("mpi_mm has started with %d tasks.\n",numtasks);
        printf("Initializing matrices...\n");

        FILE *file = fopen(argv[1], "r");   /* input file containing two matrices */

        if (file == 0)
        {
            printf("Could not open file\n");
            exit(1);
        }

        char buffer[MAXCHAR];               /* character buffer for parsing line */
        char *token;                        /* storing the token after splitting the line*/
        nra = 0;
        nca = 0;
        ncb = 0;
        bool first_done = false,            /* boolean for whether first matrix is done processing */
             second_done = false;           /* boolean for whether second matrix is done processing */

        /* just to figure out the size of matrices from file */
        while(fgets(buffer, MAXCHAR, file) != NULL)
        {
            if (strcmp(buffer, "\n") == 0 || strcmp(buffer, "\r\n") == 0)
            {
                if (first_done == false)
                {
                    first_done = true;
                }
                else
                {
                    second_done = true;
                }
            }
            else
            {
                token = strtok(buffer, ",");
                if (first_done == false)
                {
                    if (nca == 0)
                    {
                        while(token != NULL)
                        {
                            token = strtok(NULL, ",");
                            nca = nca + 1;
                        }
                    }
                    nra = nra + 1;
                }
                else if (second_done == false)
                {
                    if (ncb == 0)
                    {
                        while(token != NULL)
                        {
                            token = strtok(NULL, ",");
                            ncb = ncb + 1;
                        }
                    }
                }
            }
        }
        printf("Number of Rows of A: %i\n", nra);
        if (nra > MAXROW)
        {
            printf("Error: rows exceed %i\n", MAXROW);
            exit(1);
        }
        printf("Number of Columns of A (also Rows of B): %i\n", nca);
        if (nca > MAXCOL)
        {
            printf("Error: columns exceed %i\n", MAXCOL);
            exit(1);
        }
        printf("Number of Columns of B: %i\n", ncb);
        if (ncb > MAXCOL)
        {
            printf("Error: columns exceed %i\n", MAXCOL);
            exit(1);
        }

        double  a[nra][nca],    /* matrix A to be multiplied */
                b[ncb][nca],    /* matrix B to be multiplied (rows and columns swapped for message passing) */
                c[nra][ncb];    /* result matrix C */
        
        // Move the file pointer to the start.
        fseek(file, 0, SEEK_SET);

        first_done = false;
        second_done = false;
        i = 0;
        j = 0;

        while(fgets(buffer, MAXCHAR, file) != NULL)
        {
            if (strcmp(buffer, "\n") == 0 || strcmp(buffer, "\r\n") == 0)
            {
                if (first_done == false)
                {
                    first_done = true;
                    i = 0;
                    j = 0;
                }
                else
                {
                    second_done = true;
                }
            }
            else
            {
                token = strtok(buffer, ",");
                if (first_done == false)
                {
                    j = 0;
                    while(token != NULL)
                    {
                        a[i][j] = atof(token);
                        token = strtok(NULL, ",");
                        j += 1;
                    }
                }
                else if (second_done == false)
                {
                    j = 0;
                    while(token != NULL)
                    {
                        b[j][i] = atof(token);
                        token = strtok(NULL, ",");
                        j += 1;
                    }
                }
                i += 1;
            }
        }
        fclose(file);
        #if DEBUG
        printf("\nMatrix A:\n");
        for (i=0; i<nra; i++)
        {
            printf("\n"); 
            for (j=0; j<nca; j++) 
                printf("%6.2f   ", a[i][j]);
        }
        printf("\n******************************************************\n");

        printf("\nMatrix B:\n");
        for (i=0; i<nca; i++)
        {
            printf("\n"); 
            for (j=0; j<ncb; j++) 
                printf("%6.2f   ", b[j][i]);
        }
        printf("\n******************************************************\n");
        #endif

        /* Send matrix data to the worker tasks */
        averow = nra/numworkers;
        extra = nra%numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        for (dest=1; dest<=numworkers; dest++)
        {
            rows = (dest <= extra) ? averow+1 : averow;   	
            printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&nra, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&nca, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&ncb, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&a[offset][0], rows*nca, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&b, nca*ncb, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            offset = offset + rows;
        }

        /* Receive results from worker tasks */
        mtype = FROM_WORKER;
        for (i=1; i<=numworkers; i++)
        {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows*ncb, MPI_DOUBLE, source, mtype, 
                    MPI_COMM_WORLD, &status);
            printf("Received results from task %d\n",source);
        }
        
        #if DEBUG
        /* Print results */
        printf("******************************************************\n");
        printf("Result Matrix:\n");
        for (i=0; i<nra; i++)
        {
            printf("\n"); 
            for (j=0; j<ncb; j++) 
                printf("%6.2f   ", c[i][j]);
        }
        printf("\n******************************************************\n");
        #endif

        FILE *output = fopen("output.txt", "w+");
        for (i=0; i<nra; i++)
        {
            for (j=0; j<ncb; j++)
            {
                if (j == ncb - 1)
                    fprintf(output, "%f", c[i][j]);
                else
                    fprintf(output, "%f,", c[i][j]);
            }
            if (i < nra - 1)
                fprintf(output, "\n");
        }
        fclose(output);

        printf ("Done.\n");
        endwtime = MPI_Wtime();
        printf("wall clock time = %f\n", endwtime-startwtime);
    }


/**************************** worker task ************************************/
    if (taskid > MASTER)
    {
        mtype = FROM_MASTER;
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&nra, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&nca, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&ncb, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        double a[nra][nca];
        double b[ncb][nca];
        double c[nra][ncb];

        MPI_Recv(&a, rows*nca, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&b, nca*ncb, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

        for (k=0; k<ncb; k++)
        {
            for (i=0; i<rows; i++)
            {
                c[i][k] = 0.0;
                for (j=0; j<nca; j++)
                    c[i][k] = c[i][k] + a[i][j] * b[k][j];
            }
        }
        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&c, rows*ncb, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    }
    MPI_Finalize();
}