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


/**************************/
/* Functions about Satcnf */

bool Satcnf::parse(const char * file) {
	char buffer[512], c='\0';
	int v=0,cl=0;
	char* c_cnf;

	n=m=t=0;
	
	FILE * f=NULL;
	f=fopen(file,"ro");
	if (f==NULL) {
		perror("Cannot open input file");
		return false;
	}

	while (!feof(f)&&!ferror(f)) {
		c=fgetc(f);
	
		if (c=='c') { /* Discard comments */
            		do {
                		c=fgetc(f);
            		} while (c!='\n' && c!=EOF);
        	} 
		else if (c=='p') { /* Read the instance */
            		fscanf(f,"%s", &buffer);
			
			// Check if cnf
			if (strcmp(buffer,"cnf")!=0) {
				cerr << "Not a CNF formula" <<endl;
				fclose(f);
				return false;
			}

			// Read n and m
			fscanf(f,"%d %d",&n,&m);
			if (n<=0 || m<=0) {
				cerr << "Invalid n and m numbers" << endl;
				fclose(f);
				return false;
			}
			
			// Initalize data
			c_cnf=new char[n*m];
			for (int i=0;i<n*m;i++) c_cnf[i]='\0';
			cl=1;
			
			do {
				fscanf(f,"%d",&v);
				if (v == 0) {
					cl++;
				}
				else if ((v>0)&&(v<=n)){
					c_cnf[(v-1)*m + (cl-1)]='+';
					t++;
				}
				else if ((v<0)&&(v>=(-1*n))) {
					v*=-1;
					c_cnf[(v-1)*m + (cl-1)]='-';
					t++;
				}
			} while ((cl<=m) && (!feof(f)&&!ferror(f)));
        	}
    	}

    	fclose(f);
	
	char var;
	v=0;
	
	/* Initialize the compressed array */
	cnf = new Object[t];
	//cout << "N= " << n << ", M= " << m << ", T= " << t <<endl;
	for (int i=0;i<m;i++) {
		for (int j=0;j<n;j++) {
			if (c_cnf[j*m+i]=='+') {
				cnf[v++]=object('x',i+1,j+1);
			}
			else if (c_cnf[j*m+i]=='-') {
				cnf[v++]=object('y',i+1,j+1);
			}
		}
	}
	
	delete c_cnf;
	return true;
}
