readme.txt: This file explains the code and the structure of the algorithm.

I. Motivation

This simulator is able to simulate a family of P systems with active membranes solving SAT problem in linear time. The input of the algorithm is given by a DIMACS CNF file, which codifies an instance of SAT problem (a CNF formula) to be automatically converted to an input multiset. It is based on the stages detected during the computation of the P systems:

1. Generation.
2. Synchronization.
3. Check-out.
4. Output.


II. Installation: 

1. Install the CUDA SDK version 4.X.

2. Install the counterslib: extract the file counterslib.tar.gz inside the common folder of the CUDA SDK.

3. Copy the folder into the source folder of the CUDA SDK, and type "make".


III. Usage:

Type ./pcudaSAT -h to list the different options.
* A sequential simulation: ./pcudaSAT -s 0 -f file.cnf
* A parallel simulation on the GPU: ./pcuda -s 1 -f file.cnf
* A hybrid parallel simulation on the GPU: ./pcuda -s 2 -f file.cnf


IV. Source:

The objective of each file is the following:

main.cpp: contains the main function, calling the different algorithms.

satcnf.cpp, satcnf.h: parse the input files.

seqsolver.cpp, seqsolver.h: the sequential solvers.

object.cpp, object.cu, object.h: implementation of objects for both CPU and GPU.

evolution_division_kernel.cu, evolution_sin_d_kernel.cu, evolution_sout_d_kernel.cu, syn_check.cu, gpusolver.cu, gpusolver.h: kernels and host parts for the parallel simulator.

checkout_kernel.cu, division_kernel.cu, evolution_kernel.cu, gpuhybridsolver.cu, gpuhybridsolver.h: kernels and host parts for the hybrid parallel simulator.

/*$Id: readme.txt 2012-12-10 19:11:44 mdelamor $*/
