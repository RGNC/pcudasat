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

#include "satcnf.h"
#include "seqsolver.h"
#include "gpuhybridsolver.h"
#include "gpusolver.h"


int main (int argc, char* argv[]) {
	Satcnf scnf;
	int verbose_mode=1, solver=0;
	string input_file;
	char c='\0';
	bool sol=false;

	while ((c = getopt (argc, argv, "v:f:s:h")) != -1) {
	    switch (c) {
	        case 'v':
	            verbose_mode = atoi(optarg);
	            break;
	        case 'f':
	            input_file = optarg;
	            break;
		case 's':
		    solver = atoi(optarg);
	            break;
	        case 'h':
	        case '?':
	            cout << "Usage: pcudasat <params>, where params must be: -v (verbosity (not supported yet)) -f (input file) -s (solver: 0 sequential, 1 GPU, 2 Hybrid) -h (help)" << endl;
	  	    cout << "Version alpha 0.2" << endl;

	    	    return 0;
        	}
    	}

	if (verbose_mode>0)
	cout << "Reading file..." << endl;
	
	if (! scnf.parse(input_file.c_str()))
	     return -1;

	if (verbose_mode>0)
	cout << "[OK]" << endl << "Information" << endl;
	
	Object* o= scnf.getCNF();
	
	if (verbose_mode>0) {
	cout << "Instance size: N=" << scnf.getN() << ", M=" << scnf.getM() << endl;
	cout << "Objects: ";
	for (int i=0; i<scnf.getT(); i++)
 	    cout << get_variable(o[i]) << get_i(o[i]) << "," << get_j(o[i]) << " ";

	cout << endl; }

	switch (solver) {
	    case 0:
                sol=seq_solver(scnf.getN(),scnf.getM(),scnf.getT(),scnf.getCNF());
		break;
	    case 1:
		sol=gpu_solver(scnf.getN(),scnf.getM(),scnf.getT(),scnf.getCNF());
		break;
	    case 2:
		sol=gpuHybridSolver(scnf.getN(),scnf.getM(),scnf.getT(),scnf.getCNF());
		break;
	    default:
                cout << "Wrong solver type " << solver << ". Use -h for help" << endl;
		return 0;
	}
	
	if (verbose_mode>0)
        cout << endl << "Response of the system: The answer is " <<  boolalpha << sol << endl;

        return 0;
}
