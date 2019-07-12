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

#include "division_kernel.cu"
#include "evolution_kernel.cu"
#include "checkout_kernel.cu"

#define MAX_BLOCKS_X 32768
using namespace std;

extern "C" bool gpuHybridSolver(int N, int M, int T, Object * cnf) {
    uint * d_cnf, dev;
    cudaDeviceProp deviceProp;
    bool * d_response, response = false;
    uint numMemb = 1;
    uint maxMemb = (uint) pow(2.0, N);
    dim3 grid;
    uint blocksPerRow, rowsPerGrid;

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

    // allocate device memory 
    cutilSafeCall(cudaMalloc((void**)&d_cnf, deviceGlobalMem));
    cutilSafeCall(cudaMalloc((void**)&d_response, sizeof(bool)));

    cutilSafeCall(cudaMemcpy(d_cnf, cnf, sizeof(Object) * T, cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(d_response, &response, sizeof(bool), cudaMemcpyHostToDevice));
    
    // create and start timer
    uint timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

    grid = dim3(1);

    for (int i=0; i<N; i++) {
        division<<<grid, T>>>(d_cnf, numMemb);
        // check for any errors
        cutilCheckMsg("Kernel execution failed");

        numMemb<<=1;

        // setup execution parameters
        if (numMemb <= MAX_BLOCKS_X) {
            // We can use a 1D Grid
            blocksPerRow = numMemb;
            rowsPerGrid  = 1;
        } else {
            // We need to use a 2D Grid
            blocksPerRow = rowsPerGrid = (uint) sqrt(numMemb);

            while ((blocksPerRow * rowsPerGrid) < numMemb)
                blocksPerRow++;
        }

        grid = dim3(blocksPerRow, rowsPerGrid);

	//cout << "blocksx=" << blocksPerRow << ", y="<<rowsPerGrid<<endl;
        evolution<<<grid, T>>>(d_cnf, numMemb);
        // check for any errors
        cutilCheckMsg("Kernel execution failed");
    }

    checkOut<<<grid, T, sizeof(uint) * T>>>(d_cnf, d_response, M, numMemb);
    // check for any errors
    cutilCheckMsg("Kernel execution failed");

    // stop and destroy timer
    cutilCheckError(cutStopTimer(timer));
    cout << endl << "Execution time: " << cutGetTimerValue(timer) << " ms" << endl;
    cutilCheckError(cutDeleteTimer(timer));

    cutilSafeCall(cudaMemcpy(&response, d_response, sizeof(bool), cudaMemcpyDeviceToHost));

    //printf("\nEL RESULTADO ES: %s\n", (response)?"true":"false");

    cutilSafeCall(cudaFree(d_cnf));
    cutilSafeCall(cudaFree(d_response));

    cudaThreadExit();

    return response;
}
