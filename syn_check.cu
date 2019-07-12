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


#ifndef _SYN_CHECK_KERNEL_CU_
#define _SYN_CHECK_KERNEL_CU_

#include "object.cu"

__global__ void syn_check(Object * multiset, uint * cm1, uint N, uint M, uint numMemb) {
//	extern __shared__ uint shared[];
	
	__shared__ ushort end_d;
	__shared__ ushort end_j;
	__shared__ uint end_d2;
	__shared__ uint charge;
   	const uint tid = threadIdx.x;
    	const uint bid = blockIdx.x+gridDim.x*blockIdx.y;  
    	const uint blockSize = blockDim.x;

    	uint o = multiset[bid*blockSize+tid];
	uint d=N;
	__shared__ uint c;
    	uint i = getI(o);//(o>>8)&0xFFFF;
	uint j = getJ(o);//o&0xFF;
    	char var = getVar(o);//(char)(o>>24); 

	/* STAGE 2: SYNCHRONIZATION */
	uint cont = (var=='r');//(var>>1)&0x1; // 1 if r, 0 if null

	if (tid==0) {
		end_d=3*N-2;
		end_j=2*N-1;
		end_d2=3*N+2*M+1;
		charge='+';
		c=1;
	}

	__syncthreads();

	for (;d<=end_d;d++) {
		if (j<=end_j) j+=cont;
	}

	/* STAGE 3: CHECKOUT */
	for (;d<=end_d2;d++) {
		o='-';
		/* Send out r */
		if (i==1) {
			o=atomicExch(&charge,'-'); // reuse o
			if (o=='+') { // Only the first is sent out
				var='R';
			}
		}
		__syncthreads();
	
		/* Send in r */
		if (charge=='-') {
			if (i>0) i--;
			if (var=='R') {
				c++;
				var='r';
			}
		}
		else {
			break; // If not '-', it will never be
		}
		__syncthreads();
		charge='+';
		__syncthreads();
	}	
	
	o=setObject(var,i,j);
	multiset[bid*blockSize+tid]=o;

	if (tid==0 && c==M+1) {
		atomicAdd(cm1,1);
	}
}

#endif // _CHECKOUT_KERNEL_H_
