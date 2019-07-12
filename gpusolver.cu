/*
    pcudaSAT: Simulating an efficient solution to SAT with active membranes on the GPU 
    This simulator is published on:
    J.M. Cecilia, J.M. García, G.D. Guerrero, M.A. Martínez-del-Amor, I. Pérez-Hurtado,
    M.J. Pérez-Jiménez. Simulating a P system based efficient solution to SAT by using
    GPUs, Journal of Logic and Algebraic Programming, 79, 6 (2010), 317-325

    pcudaSAT is a subproject of PMCGPU (Parallel simulators for Membrane 
                                       Computing on the GPU)   
 
    Copyright (c) 2010 Miguel Á. Martínez-del-Amor (RGNC, University of Seville)
 		       Ginés D. Guerrero (GACOP, University of Murcia)
    
    This file is part of pcudaSAT.
  
    pcudaSAT is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    pcudaSAT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with pcudaSAT.  If not, see <http://www.gnu.org/licenses/>. */

#include <cutil_inline.h>
#include <iostream>
#include <math.h>

#include "object.h"

#include "evolution_division_kernel.cu"
#include "evolution_sout_d_kernel.cu"
#include "evolution_sin_d_kernel.cu"
#include "syn_check.cu"

#define MAX_BLOCKS_X 32768
using namespace std;


void print_gpu(Object* device_multiset,unsigned int number_membranes, unsigned int T) {
	
	Object * multiset;
	multiset = new Object[number_membranes*T];
	
	cutilSafeCall(cudaMemcpy(multiset, device_multiset, sizeof(Object) * T * number_membranes, cudaMemcpyDeviceToHost));

        cout << "Number of membranes: " << number_membranes << endl;

        cout << "Multisets: ";
        for (int i=0; i<number_membranes; i++) {
                cout << "|"<< i << "|: ";

                for (int j=0; j<T; j++) {
                        int o=i*T+j;

                        cout << get_variable(multiset[o]) << get_i(multiset[o]) << "," << get_j(multiset[o]) << " ";

                }
                if (i%8==7) cout << endl;
        }
        cout << endl;
	delete multiset;
}



extern "C" bool gpu_solver(int N, int M, int T, Object * cnf) {

	/* For the simulator */
	uint * d_cnf, dev;
    	cudaDeviceProp deviceProp;
    	bool response = false;
   	uint numMemb = 1;
	uint maxMemb = (uint) pow(2.0, N);
   	dim3 grid;
    	uint blocksPerRow, rowsPerGrid;

	/* For the P system */
	uint d=1;
	uint *d_cm1,cm1=0;
       	
	/* Initialize GPU */
	char * def_dev = getenv("DEFAULT_DEVICE");
	if (def_dev!=NULL)
		cudaSetDevice(dev= atoi(def_dev));
	else
		cudaSetDevice(dev = cutGetMaxGflopsDeviceId());
	
    	cutilSafeCall(cudaGetDeviceProperties(&deviceProp, dev));

    	uint maxDeviceMemb = deviceProp.maxGridSize[0] * deviceProp.maxGridSize[1];
    	uint deviceGlobalMem = maxMemb * T * sizeof(Object);

    	// test conditions
    	cutilCondition(maxMemb <= maxDeviceMemb);
    	cutilCondition(T <= deviceProp.maxThreadsPerBlock);
    	cutilCondition(deviceGlobalMem <= deviceProp.totalGlobalMem);
		
	// create and start timer
    	uint timer = 0;
    	cutilCheckError(cutCreateTimer(&timer));
    	cutilCheckError(cutStartTimer(timer));

	// allocate device memory 
    	cutilSafeCall(cudaMalloc((void**)&d_cnf, deviceGlobalMem));
    	cutilSafeCall(cudaMalloc((void**)&d_cm1, sizeof(uint)));

    	cutilSafeCall(cudaMemcpy(d_cnf, cnf, sizeof(Object) * T, cudaMemcpyHostToDevice));
    	cutilSafeCall(cudaMemcpy(d_cm1, &cm1, sizeof(uint), cudaMemcpyHostToDevice));
    
    	grid = dim3(1);

	/* STAGE 1: GENERATION */
	d=1;

        do {
                evolution_division<<<grid, T>>>(numMemb,N,d_cnf);
                cutilCheckMsg("Kernel execution failed");

		numMemb<<=1;
		// setup execution parameters
        	if (numMemb <= MAX_BLOCKS_X) {
            		// We can use a 1D Grid
            		blocksPerRow = numMemb;
            		rowsPerGrid  = 1;
        	} else {
			// We need to use 2D Grid
            		//blocksPerRow = MAX_BLOCKS_X;
            		//rowsPerGrid = numMemb/MAX_BLOCKS_X;
			blocksPerRow = rowsPerGrid = (uint) sqrt(numMemb);

            		while ((blocksPerRow * rowsPerGrid) < numMemb)
                		blocksPerRow++;
        	}
		grid = dim3(blocksPerRow, rowsPerGrid);

	        //cout << "Running numMembranes=" << numMemb << ", maxgridx=" << deviceProp.maxGridSize[0] << ", blocksx=" << blocksPerRow << ", y="<<rowsPerGrid<<endl;

                evolution_sout_d<<<grid, T>>>(numMemb,d_cnf);
		cutilCheckMsg("Kernel execution failed");

		if (d<N) {
			evolution_sin_d<<<grid, T>>>(numMemb,N,d_cnf);
        	        cutilCheckMsg("Kernel execution failed");
		}

                d++;
        } while (d<=N);

	d--;

	//print_gpu(d_cnf,numMemb,T);

	syn_check<<<grid,T>>>(d_cnf,d_cm1,N,M,numMemb);
    	// check for any errors
    	cutilCheckMsg("Kernel execution failed");

	d=3*N-1;
	ushort e=1;
	uint c=1;

	//print_gpu(d_cnf,numMemb,T);
	cutilSafeCall(cudaMemcpy(&cm1, d_cm1, sizeof(uint), cudaMemcpyDeviceToHost));

	response=(cm1>0);
		
    	// stop and destroy timer
    	cutilCheckError(cutStopTimer(timer));
    	cout << endl << "Execution time: " << cutGetTimerValue(timer) << " ms" << endl;
    	cutilCheckError(cutDeleteTimer(timer));

    	//printf("\nEL RESULTADO ES: %s\n", (response)?"true":"false");

    	cutilSafeCall(cudaFree(d_cnf));
    	cutilSafeCall(cudaFree(d_cm1));

    	cudaThreadExit();

    	return response;
}
