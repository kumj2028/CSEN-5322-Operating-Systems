#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
 
int main (int argc, char* argv[])
{
  int rank, size, i, j;
 
  MPI_Init (&argc, &argv);                      /* Initialization step to start MPI */
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);        /* get the id of the current process */
  MPI_Comm_size (MPI_COMM_WORLD, &size);        /* get the number of processes involved */
   
  srand((int)time(NULL)); 
   
  int r = rand() % 20000;
 
 
  double y = r* sin((size-rank+1)*r);

  r = abs(y); 

   for (i=0; i<=r; i++)
   
     {
        j=i*i*i;
     }


   printf( "Hello world from process %d of %d\n", rank, size);

   MPI_Finalize();                               /* Terminates MPI execution environment */
  
  return 0;
}