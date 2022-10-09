mpicc mpi_hello.c -o hello
mpirun -np 4 ./hello

or,  // if your math function is causing problem, use -lm. "lm" is to include the math library.
 mpicc mpi_hello.c -o hello -lm
 mpirun -np 4 ./hello


The expected outputs are as following in any order:
Hello world from process 0 of 4
Hello world from process 1 of 4
Hello world from process 2 of 4
Hello world from process 3 of 4
