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

#ifndef _EVOLUTION_SOUT_KERNEL_CU_
#define _EVOLUTION_SOUT_KERNEL_CU_

#include "object.cu"

/*
__device__ inline void object(Object & o, char var, short int i, short int j) {
	o=(var)<<24;
    	o=o|((i)<<8);
    	o=o|(j&0xFF);
}

__device__ inline char getVar(Object & o) {
    	return (char)(o>>24);
}

__device__ inline ushort getJ(Object & o) {
    	return (ushort) (o&0xFF);
}

__device__ inline ushort getI(Object &o) {
	return (short int) ((o>>8)&0xFFFF);
} */

__global__ static void evolution_sout_d(const uint numMemb, Object * cnf) {
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
	ushort i = getI(o);

	if ((bid&pivot)!=0) { // Charge = '+'
        	if (var == 'x' && j==1) {
                	var='r';
                }
		else if (var != 'r' && var != 0) {
			j--;
			if (var == 'y' && j==0) {
				var=0;
				i=0;
			}
		}
	}
        else {   // Charge = '-'
                if (var == 'y' && j==1) {
                        var='r';
                }
                else if (var != 'r' && var != 0) {
                        j--;
                        if (var == 'x' && j==0) {
                                var=0;
                                i=0;
                        }
                }
        }
	__syncthreads();


	o=setObject(var,i,j);

    	cnf[bid*blockSize+tid] = o;
}

#endif // _EVOLUTION_KERNEL_H_
