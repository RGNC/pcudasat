# Parallel simulation of a family of P systems with active membranes solving SAT in linear time (PCUDASAT) #

----------
## 1. Description ##

The project PCUDASAT is a continuation of PCUDA. It aims to develop optimized simulators for a family of P systems with active membranes solving SAT in linear time. A sequential and two parallel (CUDA based) simulators are included. Their simulation algorithm is based on the stages that can be identified in the computation of any P system in the family: Generation, Synchronization, Check-out and Output. The code is tailored to the them, saving space in the definition of, for example, auxiliary objects.

The simulators receive as input a DIMACS CNF file, which codifies an instance of SAT through a CNF formula. The output is a summary of the codification, and the answer: yes or not. Therefore, they merely act as a SAT solver based on a P systems based solution. 

The CUDA simulator design is similar to the one used in PCUDA: it assigns a thread block to each elementary membrane (which is each truth valuation to the CNF formula). However, the number of objects to be represented inside each membrane in memory has been decreased. In this case, it is enough to store only the objects appearing in the input multiset (which is a literal of the CNF formula), so that the rules can evolve them, one by one. Therefore, the CUDA design assigns a thread to each object of the input multiset.
A second CUDA simulator was developed to optimize the usage of GPU resources. It is a hybrid solution, inasmuch as it reproduces the execution of the first stages, but the last ones are more efficiently executed on the GPU.

Although the PCUDASAT simulators are less flexible, the performance and scalability has been much increased compared to PCUDA simulators, for this special case study.

It is noteworthy to mention further simulators developed to be better tailored to the GPU idiosyncrasy. Better optimization strategies were performed, and newer GPU cards, multi-GPU and supercomputers were utilized (**not provided here yet**).

----------
## 2. Installation and Usage ##

### 2.1. Installation: 

  1. Install the CUDA SDK version 4.X.

  2. Install the counterslib: extract the file counterslib.tar.gz inside the common folder of the CUDA SDK.

  3. Copy the folder into the source folder of the CUDA SDK, and type "make".

### 2.2. Usage:

Type ./pcudaSAT -h to list the different options.
  * A sequential simulation: ./pcudaSAT -s 0 -f file.cnf
  * A parallel simulation on the GPU: ./pcuda -s 1 -f file.cnf
  * A hybrid parallel simulation on the GPU: ./pcuda -s 2 -f file.cnf

----------
## 3. Publications ##

### 3.1. Journals ###

* José M. Cecilia, José M. García, Ginés D. Guerrero, Miguel A. Martínez-del-Amor, Mario J. Pérez-Jiménez, Manuel Ujaldón. **The GPU on the Simulation of Cellular Computing Models**, *Soft Computing*, 16, 2 (2012), 231-246.
* José M. Cecilia, José M. García, Ginés D. Guerrero, Miguel A. Martínez-del-Amor, Ignacio Pérez-Hurtado, Mario J. Pérez-Jiménez. **Simulating a P system based efficient solution to SAT by using GPUs**, *Journal of Logic and Algebraic Programming*, 19, 6 (2010), 317-325.

### 3.2. Conference Contributions ###

* José M. Cecilia, José M. García, Ginés D. Guerrero, Miguel A. Martínez-del-Amor, Mario J. Pérez-Jiménez, Manuel Ujaldón. **P Systems Simulations on Massively Parallel Architectures**, *19th Intl. Conference on Parallel Architectures and Compilation Techniques (PACT'10)*, September 2010, Viena, Austria, (2010).
* José M. Cecilia, Ginés D. Guerrero, José M. García, Miguel Á. Martínez-del-Amor, Mario J. Pérez-Jiménez, Manuel Ujaldón. **Enhancing the Simulation of P Systems for the SAT Problem on GPUs**, *Symposium on Application Accelerators in High Performance Computing*, July 2010, Knoxville, USA, (2010).

### 3.3 Ph.D. Thesis ###

* Miguel Á. Martínez-del-Amor. [Accelerating Membrane Systems Simulators using High Performance Computing with GPU.](http://www.cs.us.es/~mdelamor/research.html#thesis) May 2013, University of Seville. Advised by Mario J. Pérez-Jiménez and Ignacio Pérez-Hurtado.
* José M. Cecilia. The GPU as a Processor for Novel Computation: Analysis and Contributions. 2011, University of Murcia. Advised by J.M. García and M. Ujaldón.

----------
## 4. Downloads ##

[Link to PCUDASAT Files](http://sourceforge.net/projects/pmcgpu/files/PCUDASAT/)

[Required Counterslib library](http://sourceforge.net/projects/pmcgpu/files/counterslib)

Read the howto.pdf (extract from Miguel A. Martínez-del-Amor's thesis) for futher information about the simulators. You can find it in the [root folder of files of PMCGPU](http://sourceforge.net/projects/pmcgpu/files).

----------
## 5. How to acknowledge ##

If you intend to create a branch of PCUDASAT, or use its produced results, please consider citing the following publications:

*José M. Cecilia, José M. García, Ginés D. Guerrero, Miguel A. Martínez-del-Amor, Mario J. Pérez-Jiménez, Manuel Ujaldón. The GPU on the Simulation of Cellular Computing Models, Soft Computing, 16, 2 (2012), 231-246.*

*José M. Cecilia, José M. García, Ginés D. Guerrero, Miguel A. Martínez-del-Amor, Ignacio Pérez-Hurtado, Mario J. Pérez-Jiménez. Simulating a P system based efficient solution to SAT by using GPUs, Journal of Logic and Algebraic Programming, 19, 6 (2010), 317-325.*

----------
## 6. Funding ##

This work has been jointly supported by the Fundación Séneca (Agencia Regional de Ciencia y Tecnología,
Región de Murcia) under grant 00001/CS/2007, by the Spanish MICINN under grants TIN2009-13192 and TIN2009-14475-C04, by the European Commission FEDER funds under grant Consolider Ingenio-2010 CSD2006-00046, and by the Junta of Andalucia of Spain under projects P06-TIC02109 and P08-TIC04200. 
