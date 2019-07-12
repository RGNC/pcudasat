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

#ifndef _EVOLUTION_DIVISION_KERNEL_CU_
#define _EVOLUTION_DIVISION_KERNEL_CU_

#include "object.cu"

__global__ static void evolution_division(const uint numMemb,const uint N,Object* cnf) {
	const uint bid = blockIdx.x+gridDim.x*blockIdx.y;    
    	const uint tid = threadIdx.x;
    	const uint numBlocks = numMemb;
    	const uint blockSize = blockDim.x;
	uint o=0;
	char var;
	short int j;

	o=cnf[bid*blockSize+tid];
	var = getVar(o);
	j= getJ(o);

    	if (bid >= numMemb)
        	return;
	
	if (var == 'r' && j<=2*N-1) {
		j++;
		o=setJ(o,j);
		cnf[bid*blockSize+tid]=o;
	}
	__syncthreads();


    	cnf[blockSize*(numBlocks+bid)+tid] = o;


}

#endif // _DIVISION_KERNEL_H_
