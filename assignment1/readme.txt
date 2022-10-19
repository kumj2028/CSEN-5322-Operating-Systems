To run the first part of the assignment do:

mpicc matrixmul.c -o matrixmul
mpirun -np 4 ./matrixmul input0.txt

where "-np 4" specifies number of processes (needs at least 2 since needs at least 1 master and 1 worker)
and "input0.txt" can be any file containing two matrices.
"output.txt" is where the resulting matrix is written.

To run the second part of the assignment do:

mpicc matmulperform.c -o matmulperform

For serial execution do
mpirun -np 2 ./matmulperform

For parallel execution do
mpirun -np 4 ./matmulperform

where "-np 4" can be any integer > 3.
"output.csv" is where the times for the execution is written.