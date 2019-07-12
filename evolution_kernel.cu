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

#ifndef _EVOLUTION_KERNEL_CU_
#define _EVOLUTION_KERNEL_CU_

#include "object.cu"

/*

__device__ inline void setVar(Object & o, char var) {
    o=(o&0xFFFFFF)|(var<<24);
}

__device__ inline void setJ(Object & o, ushort j) {
    o=(o&0xFFFFFF00)|j;
}

__device__ inline char getVar(Object & o) {
    return (char)(o>>24);
}

__device__ inline ushort getJ(Object & o) {
    return (ushort) (o&0xFF);
}*/

__global__ static void evolution(Object * cnf, const uint numMemb) {
    const uint bid = blockIdx.x+gridDim.x*blockIdx.y;      
    const uint tid = threadIdx.x;
    const uint blockSize = blockDim.x;
    uint pivot = numMemb;

    if (bid >= pivot)
        return;

    pivot >>= 1;

    Object o = cnf[bid*blockSize+tid];
    char var = getVar(o);
    ushort j = getJ(o);

    if (((var == 'x') && ((bid&pivot) == 0)) || ((var == 'y') && ((bid&pivot) != 0)))
        if (j == 1)
	    o=setVar(o, 'r');
        else
	    o=setJ(o, j-1);
    else if ((var == 'x') || (var == 'y'))
        if (j == 1)
            o=setVar(o, 'H');
        else
            o=setJ(o, j-1);

    cnf[bid*blockSize+tid] = o;
}

#endif // _EVOLUTION_KERNEL_H_
