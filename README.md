# Sw4lite RAJA
**Sw4lite** is a bare bone version of [SW4](https://geodynamics.org/cig/software/sw4) ([Github](https://github.com/geodynamics/sw4)) intended for testing performance optimizations in a few
important numerical kernels of SW4.

To build
--------
1. git clone and install  RAJA 0.7.0 ( https://github.com/LLNL/RAJA/releases )
   1.1 On machines with GPUs install RAJA with CUDA enabled and OpenMP disabled
   1.2 On CPU only machines do a default install of RAJA ( usesOpenMP)
2. Edit the makefile and point RAJA_LOCATION to install above
3. On Summit : ml cuda netlib-lapack/3.8.0
4. Compile using : 
   4.1 CUDA:  
```
   make ckernel=yes openmp=no raja=yes -j
```
   4.2 OpenMP: 
```
   make ckernel=yes openmp=yes raja=yes -j 
```


To run
------

CUDA Version:

The CUDA version has only been tesed on CORAL and CORAL-EA systems with Power8/9 processors and Nvidia P100/V100 GPUs
with Unified Memory

On LLNL machines :
```
   lrun -T4 ./sw4lite <input_file>
```
On ORNL machines use: 
```
jsrun -n <number of ranks = num_nodes *6 > -g1 -a1 -c7 ./sw4lite <input_file>
```

OpenMP Version:

To run sw4lite with OpenMP threading, you need to assign the number of threads per
MPI-task by setting the environment variable OMP_NUM_THREADS, e.g.,
```
setenv OMP_NUM_THREADS 4
```
An example input file is provided under `tests/pointsource/pointsource.in`. This case solves the
elastic wave equation for a single point source in a whole space or a half space. The input file is
given as argument to the executable, as in the example:
```
mpirun -np 16 sw4lite pointsource.in
```
Output from a run is provided at `tests/pointsource/pointsource.out`.
For this point source example, the analytical solution is known. The error is printed at the end:
```
Errors at time 0.6 Linf = 0.569416 L2 = 0.0245361 norm of solution = 3.7439
```
When modifying the code, it is important to verify that these numbers have not changed.

Some timings are also output. The average execution times (in seconds) over all MPI processes are reported as follows:
1. Total execution time for the time stepping loop,
2. Communication between MPI-tasks (BC comm)
3. Imposing boundary conditions (BC phys),
4. Evaluating the difference scheme for divergence of the stress tensor (Scheme),
5. Evaluating supergrid damping terms (Supergrid), and
6. Evaluating the forcing functions (Forcing)

The code under `tests/testil` is a stand alone single-core program that only exercises the computational kernel (Scheme).
