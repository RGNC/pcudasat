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

#ifndef __SATCNF
#define __SATCNF

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <string>
#include <sstream>

#include "object.h"

using namespace std;

class Satcnf {
private:
	unsigned int n;
	unsigned int m;
	unsigned int t;

	Object * cnf;

public:
	Satcnf () {n=m=t=0; cnf=NULL;}
	~Satcnf () { if (cnf!=NULL) delete cnf; }

	/** Create a CNF instance from the input file
	*/
	bool parse(const char * file);

	/** Get number of variables
	*/
	unsigned int getN() { return n;}

	/** Get number of clauses
	*/
	unsigned int getM() { return m;}

	/** Get length of the formula
	*/
	unsigned int getT() { return t;}

	/** Get the FNC formula in compress format:
		1 Byte=X or Y
		2 Bytes=clause
		1 Byte=variable
	*/
	Object * getCNF() { return cnf; }
};

#endif
