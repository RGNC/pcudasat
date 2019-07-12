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

#ifndef _CHECKOUT_KERNEL_CU_
#define _CHECKOUT_KERNEL_CU_

#include "object.cu"

__global__ static void checkOut(const Object * cnf, bool * response, const uint numClauses, const uint numMemb) {
    extern __shared__ int shared[];
    __shared__ bool result;

    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x+gridDim.x*blockIdx.y;  
    const uint blockSize = blockDim.x;
    const uint o = cnf[bid*blockSize+tid];
    uint i = (o>>8)&0xFFFF;
    const char var = (char)(o>>24); 

    if (bid >= numMemb)
        return;

    if (var != 'r')
	i = 0;

    shared[tid] = 0;
    
    if (tid == 0)
        result = true;

    __syncthreads();

    if (i != 0)
        shared[i-1] = 1;

    __syncthreads();

    if (tid < numClauses) 
        if (shared[tid] != 1)
            result = false;

    __syncthreads();

    if ((tid == 0) && (result == true))
        *response = result;
}

#endif // _CHECKOUT_KERNEL_H_
