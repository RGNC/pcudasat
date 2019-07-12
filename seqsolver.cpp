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

#include "seqsolver.h"
#include <iostream>
#include <timestat.h>
using namespace std;
/********************/
/* GLOBAL VARIABLES */
int d=0;
int c=0;
int e=0;

bool size_exeeced (long unsigned int number_membr, unsigned int T) {
	return (number_membr * T) + number_membr >= 3.5*1024*1024*1024;
}

void print(Object* multiset, unsigned int number_membranes, unsigned int T) {
	cout << "Number of membranes: " << number_membranes << ", d" << d << ", c" << c << endl;

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
}


/*************************/
/** STAGE 1: GENERATION **/
/**	Divide every membrane to a new position and performs evolution for "r" objects. 
	Returns the new number of membranes
*/
unsigned int evolution_division (long unsigned int NM, unsigned int N, unsigned int T, Object * multiset) {
	char var='\0';
	Object cur=0;
	short int i,j;
	unsigned int pos=0,end=T-1;

	// FOR EVERY MEMBRANE
	for (long unsigned int m=0; m<NM; m++) {
		pos=0;
		end=T-1;

		// FOR EVERY OBJECT
		for (unsigned int o=0;o<T;o++) {
			cur=multiset[m*T+o];
			var=get_variable(cur);
			// COPY THE OBJECTS AT THE BEGINNING
			// OF THE NEW MEMBRANE
			if (var!=0) {
				if (var=='r') {
					i=get_i(cur); j=get_j(cur);
					if (j<=2*N-1) {
						j++;
						cur=object(var,i,j);
						multiset[m*T+o]=cur;
					}
				}
				multiset[(m+NM)*T + (pos++)]= cur;
			}
			// APPEND AT THE END THE EMPTY OBJECTS
			else {
				multiset[(m+NM)*T + (end--)]= cur;
			}
		}
	}

	return NM*2;
}

/**	Perform de evolutions when the membranes has charges and d is sending out
*/
void evolution_sout_d(unsigned int cur_membr,unsigned int thresold,unsigned int T,Object * multiset) {
	Object cur;
	char var,charge;
	short int i,j;

	if (cur_membr < thresold) charge='+';
	else charge='-';

	for (unsigned int it=0;it<T;it++) {
		cur=multiset[cur_membr*T+it];
		var=get_variable(cur);
		i=get_i(cur);
		j=get_j(cur);
		
		if (charge=='+'){
			if (var == 'x' && j==1) {
				var='r';
			}
			else if (var =='x') {
				j--;
			}
			else if (var == 'y' && j==1) {
				var=0;
				i=j=0;
			}
			else if (var == 'y') {
				j--;
			}
		}
		else {
			if (var == 'x' && j==1) {
				var=0;
				i=j=0;
			}
			else if (var =='x') {
				j--;
			}
			else if (var == 'y' && j==1) {
				var='r';
			}
			else if (var == 'y') {
				j--;
			}
		}

		multiset[cur_membr*T+it]=object(var,i,j);
	}
}

/**	Perform the evolutions when the d is sending in again to the membranes
*/
void evolution_sin_d(unsigned int cur_membr,unsigned int N,unsigned int T,Object * multiset) {
	Object cur;
	char var,charge;
	short int i,j;

	for (unsigned int it=0;it<T;it++) {
		cur=multiset[cur_membr*T+it];
		var=get_variable(cur);
		j=get_j(cur);
		if (var == 'r') {
			if (j<=2*N-1) j++;
			i=get_i(cur);
			multiset[cur_membr*T+it]=object(var,i,j);
		}
	}
	

}

/**	Performs the Generation Stage
*/
void generation (unsigned int N,unsigned int M,unsigned int T,unsigned int num_membr, Object * multiset) {
	unsigned int num_membr_cur=1;

	d=1;
	
	while (d <= N-1) {
		num_membr_cur = evolution_division(num_membr_cur,N,T,multiset);
		//print (multiset,num_membr_cur,T);

		for (unsigned int i=0; i < num_membr_cur; i++) {
			evolution_sout_d(i,num_membr_cur/2,T,multiset);
			evolution_sin_d(i,N,T,multiset);
		}
		d++;
	}

	num_membr_cur=evolution_division(num_membr_cur,N,T,multiset);

	for (unsigned int i=0; i < num_membr_cur; i++) {
		evolution_sout_d(i,num_membr_cur/2,T,multiset);
	}
	//print (multiset,num_membr_cur,T);
}


/********************************/
/*** STAGE 2: SYNCHRONIZATION ***/
/**	Performs evolutions to r objects
	and compact the elements of the multiset deleting null objects
*/
void evolution_r_compact(unsigned int N,unsigned int T,unsigned int num_membr,Object* multiset) {
	unsigned int last=0;
	char var;
	short int i,j;
	Object cur;

	for (int m=0;m<num_membr;m++) {
		last=0;
		for (int o=0;o<T;o++) {
			cur=multiset[m*T+o];
			var=get_variable(cur);
			if (var!=0){
				j=get_j(cur);
				if (j<=2*N-1) {
					j++;
					i=get_i(cur);
					cur=object(var,i,j);
				}
				multiset[m*T+o]=0;
				multiset[m*T+(last++)]=cur;
			}
		}
	}	
}

/**	Performs evolutions to r objects
	Assumes that the objects are compacted
*/
void evolution_r(unsigned int N,unsigned int T,unsigned int num_membr,Object* multiset) {
	char var;
	short int i,j;
	Object cur;

	for (int m=0;m<num_membr;m++) {
		for (int o=0;o<T;o++) {
			cur=multiset[m*T+o];
			var=get_variable(cur);
			if (var=='r'){
				j=get_j(cur);
				if (j<=2*N-1) {
					j++;
					i=get_i(cur);
					multiset[m*T+o]=object(var,i,j);
				}
			}
			else break;
		}
	}
}

/**	Performs the Synchronization Stage
*/
void synchronization(unsigned int N,unsigned int T,unsigned int num_membr,Object* multiset) {

	/* 1 computation step */
	evolution_r_compact(N,T,num_membr,multiset);
	d++;

	/* Computation steps */
	for (;d<=3*N-3;d++) {
		evolution_r(N,T,num_membr,multiset);
	}

	/* 1 computation step */
	evolution_r(N,T,num_membr,multiset);
	d++;
	c=1;
	e=1;

	//print (multiset,num_membr,T);
}


/***************************************/
/*** STAGE 3 and 4: Checking & Output***/
/**	Performs the send out of objects r with i=1
	Returns the charge of the membrane (if the send_out has been performed)
*/
char send_out_r1(unsigned int cur_membr,unsigned int N,unsigned int T,Object * multiset) {
	char var;
	short int i,j;
	Object cur;


	for (int o=0;o<T;o++) {
		cur=multiset[cur_membr*T+o];
		var=get_variable(cur);
		if (var=='r'){
			i=get_i(cur);
			if (i==1) {
				var='R'; // Indicates that r is in the skin
				j=get_j(cur);
				multiset[cur_membr*T+o]=object(var,i,j);
				return '-';
			}
		}
		else break;
	}

	return '+';

}

void evolution_sendin_r1(unsigned int cur_membr,char charge,unsigned int N,unsigned int T,Object * multiset) {
	char var;
	short int i,j;
	Object cur;

	if (charge == '+') return;

	c++;

	for (int o=0;o<T;o++) {
		cur=multiset[cur_membr*T+o];
		var=get_variable(cur);
		if (var=='r'){
			i=get_i(cur);
			if (i>0) {
				i--;
				j=get_j(cur);
				multiset[cur_membr*T+o]=object(var,i,j);
			}
		}
		else if (var=='R') {
			j=get_j(cur);
			multiset[cur_membr*T+o]=object('r',0,j);
		}
		else break;
	}

}

bool checking_output(unsigned int N,unsigned int M,unsigned int T,unsigned int num_membr,Object* multiset) {
	
	int di=d;
	int c_prev=c;
	int cm1=0,cm2=0,t=0,ts=0,yes=0,no=0;
	char charge='-';

	for (int m=0;m<num_membr;m++){
		c=c_prev;
		charge='-';
		for (di=d;di<=3*N+2*M+1;di++) { // actually 2*m steps
			if (charge=='+') continue; // Optimization, when no '-', it will never be '-'

			charge=send_out_r1(m,N,T,multiset);
			evolution_sendin_r1(m,charge,N,T,multiset);
			if (c==M+1) {
				cm1++;
				break;
			}
		}
	}
	
	if (cm1>0) {
		cm2=cm1; t=cm1; cm1=0;
		ts=1; t--;
		yes=1; cm2--;
		//cout << "Objects in the skin: t, yes" << endl;
	}
	else {
		d++;
		no=1;
		//cout << "Objects in the skin: no" << endl;
	}

	return yes==1;
}

bool seq_solver(int N, int M, int T, Object * cnf) {

	long int num_membr=(long int)pow(2,N);
	bool sol;
        //struct timeval tini, tfin;
	double time;
	
	//start_timer();
	
	Object * multiset = new Object[num_membr*T];

	for (int i=0; i<T; i++) {
		multiset[i]=cnf[i];
	}

	//gettimeofday(&tini, NULL);
	start_timer();
	/* STAGE 1 */
	generation(N,M,T,num_membr,multiset);
	/* STAGE 2 */
	synchronization(N,T,num_membr,multiset);
	/* STAGE 3 */
	sol=checking_output(N,M,T,num_membr,multiset);
        //gettimeofday(&tfin, NULL);

        //tiempo= (tfin.tv_sec - tini.tv_sec)*1000000 + tfin.tv_usec - tini.tv_usec;
	time=end_timer();
        
        cout << endl << "Execution time: " << time << " ms" << endl;

	delete multiset;

	return sol;
}

